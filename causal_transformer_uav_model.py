import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_causal_mask(t_query, t_key, device):
    """Build causal mask for attention.

    Args:
        t_query: Query length Tq.
        t_key: Key length Tk.
        device: Torch device.

    Returns:
        Bool tensor with shape (Tq, Tk), True where j > i (masked).
    """
    i = torch.arange(t_query, device=device).unsqueeze(1)
    j = torch.arange(t_key, device=device).unsqueeze(0)
    return j > i


class SharedRelativePositionalEncoding(nn.Module):
    """Shared trainable causal relative positional encoding.

    Relative distances are clipped to [-lmax, 0], shared across:
    - heads
    - blocks
    - streams
    - self-/cross-attention

    Args:
        lmax: Maximum absolute look-back lag.
        d_qkv: Per-head key/value dimension.

    Parameters:
        rel_k: (lmax + 1, d_qkv)
        rel_v: (lmax + 1, d_qkv)
    """

    def __init__(self, lmax, d_qkv):
        super().__init__()
        self.lmax = int(max(0, lmax))
        self.d_qkv = int(d_qkv)

        self.rel_k = nn.Parameter(torch.zeros(self.lmax + 1, self.d_qkv))
        self.rel_v = nn.Parameter(torch.zeros(self.lmax + 1, self.d_qkv))

        nn.init.normal_(self.rel_k, mean=0.0, std=0.02)
        nn.init.normal_(self.rel_v, mean=0.0, std=0.02)

    def relative_index(self, t_query, t_key, device):
        """Return clipped relative index matrix.

        Args:
            t_query: Query length Tq.
            t_key: Key length Tk.
            device: Torch device.

        Returns:
            Long tensor idx with shape (Tq, Tk),
            idx = clip(j-i, -lmax, 0) + lmax, values in [0, lmax].
        """
        i = torch.arange(t_query, device=device).unsqueeze(1)
        j = torch.arange(t_key, device=device).unsqueeze(0)
        rel = j - i
        rel = torch.clamp(rel, min=-self.lmax, max=0)
        idx = rel + self.lmax
        return idx.long()

    def gather_rel_k(self, t_query, t_key, device, dtype):
        """Gather relative key embeddings.

        Returns:
            Tensor a_k with shape (Tq, Tk, d_qkv).
        """
        idx = self.relative_index(t_query, t_key, device)
        return self.rel_k[idx].to(dtype=dtype)

    def gather_rel_v(self, t_query, t_key, device, dtype):
        """Gather relative value embeddings.

        Returns:
            Tensor a_v with shape (Tq, Tk, d_qkv).
        """
        idx = self.relative_index(t_query, t_key, device)
        return self.rel_v[idx].to(dtype=dtype)


class MaskedRelativeMultiHeadAttention(nn.Module):
    """Masked multi-head attention with shared trainable relative PE.

    Attention equation per head:
        alpha_ij = softmax_j( q_i^T (k_j + a^K_ij) / sqrt(d_qkv) )
        out_i = sum_j alpha_ij (v_j + a^V_ij)

    Args:
        d_model: Input model width.
        n_heads: Number of heads.
        d_qkv: Per-head q/k/v width.
        attn_dropout: Dropout on attention weights.
        use_out_proj: Whether to apply post-concat output projection.
        rel_pos: Shared relative positional module.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        d_qkv,
        attn_dropout,
        use_out_proj,
        rel_pos,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_qkv = int(d_qkv)
        self.inner_dim = self.n_heads * self.d_qkv
        self.use_out_proj = bool(use_out_proj)

        self.rel_pos = rel_pos

        self.q_proj = nn.Linear(self.d_model, self.inner_dim)
        self.k_proj = nn.Linear(self.d_model, self.inner_dim)
        self.v_proj = nn.Linear(self.d_model, self.inner_dim)

        if self.use_out_proj:
            self.out_proj = nn.Linear(self.inner_dim, self.d_model)
        else:
            if self.inner_dim != self.d_model:
                raise ValueError(
                    "When use_out_proj=False, n_heads*d_qkv must equal d_model. "
                    f"Got {self.inner_dim} vs {self.d_model}."
                )
            self.out_proj = None

        self.attn_dropout = nn.Dropout(float(attn_dropout))

    def forward(self, q_in, k_in, v_in, return_attn=False):
        """Compute masked relative attention.

        Args:
            q_in: (B, Tq, d_model)
            k_in: (B, Tk, d_model)
            v_in: (B, Tk, d_model)
            return_attn: Return attention weights if True.

        Returns:
            out: (B, Tq, d_model)
            attn(optional): (B, H, Tq, Tk)
        """
        if q_in.dim() != 3 or k_in.dim() != 3 or v_in.dim() != 3:
            raise ValueError("Expected rank-3 inputs for q/k/v.")
        if k_in.size(1) != v_in.size(1):
            raise ValueError("k/v sequence length mismatch.")
        if k_in.size(0) != q_in.size(0) or v_in.size(0) != q_in.size(0):
            raise ValueError("Batch size mismatch for q/k/v.")

        bsz, t_query, _ = q_in.shape
        t_key = k_in.size(1)

        q = self.q_proj(q_in).view(bsz, t_query, self.n_heads, self.d_qkv).transpose(1, 2)
        k = self.k_proj(k_in).view(bsz, t_key, self.n_heads, self.d_qkv).transpose(1, 2)
        v = self.v_proj(v_in).view(bsz, t_key, self.n_heads, self.d_qkv).transpose(1, 2)

        score_content = torch.einsum("bhid,bhjd->bhij", q, k)

        rel_k = self.rel_pos.gather_rel_k(t_query, t_key, q.device, q.dtype)
        rel_v = self.rel_pos.gather_rel_v(t_query, t_key, q.device, q.dtype)

        score_rel = torch.einsum("bhid,ijd->bhij", q, rel_k)
        logits = (score_content + score_rel) / math.sqrt(float(self.d_qkv))

        causal_mask = _build_causal_mask(t_query, t_key, logits.device)
        neg_large = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), neg_large)

        logits = logits - logits.max(dim=-1, keepdim=True).values
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)

        context_content = torch.einsum("bhij,bhjd->bhid", attn, v)
        context_rel = torch.einsum("bhij,ijd->bhid", attn, rel_v)
        context = context_content + context_rel

        out = context.transpose(1, 2).contiguous().view(bsz, t_query, self.inner_dim)
        if self.out_proj is not None:
            out = self.out_proj(out)

        if return_attn:
            return out, attn
        return out


class StreamFFN(nn.Module):
    """Position-wise FFN used in each stream.

    Input/Output:
        (B, T, d_model) -> (B, T, d_model)
    """

    def __init__(self, d_model, ff_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(d_model), int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        return self.net(x)


class CausalTransformerMultiInputBlock(nn.Module):
    """One 3-stream masked causal transformer block.

    Inputs:
        x, a, y: (B, T, d_model)

    Outputs:
        x_out, a_out, y_out: (B, T, d_model)
    """

    def __init__(
        self,
        d_model,
        n_heads,
        d_qkv,
        ff_dim,
        dropout,
        attn_dropout,
        use_out_proj,
        rel_pos,
    ):
        super().__init__()

        self.resid_dropout = nn.Dropout(float(dropout))

        def make_attn():
            return MaskedRelativeMultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                d_qkv=d_qkv,
                attn_dropout=attn_dropout,
                use_out_proj=use_out_proj,
                rel_pos=rel_pos,
            )

        # Self-attention per stream.
        self.self_x = make_attn()
        self.self_a = make_attn()
        self.self_y = make_attn()

        # Cross-attention per stream pair.
        self.x_from_a = make_attn()
        self.x_from_y = make_attn()
        self.a_from_x = make_attn()
        self.a_from_y = make_attn()
        self.y_from_x = make_attn()
        self.y_from_a = make_attn()

        # LayerNorms for self stage.
        self.ln_x_self = nn.LayerNorm(int(d_model))
        self.ln_a_self = nn.LayerNorm(int(d_model))
        self.ln_y_self = nn.LayerNorm(int(d_model))

        # LayerNorms for cross stage.
        self.ln_x_from_a = nn.LayerNorm(int(d_model))
        self.ln_x_from_y = nn.LayerNorm(int(d_model))
        self.ln_a_from_x = nn.LayerNorm(int(d_model))
        self.ln_a_from_y = nn.LayerNorm(int(d_model))
        self.ln_y_from_x = nn.LayerNorm(int(d_model))
        self.ln_y_from_a = nn.LayerNorm(int(d_model))

        # FFN per stream.
        self.ffn_x = StreamFFN(d_model, ff_dim, dropout)
        self.ffn_a = StreamFFN(d_model, ff_dim, dropout)
        self.ffn_y = StreamFFN(d_model, ff_dim, dropout)

        self.ln_x_ffn = nn.LayerNorm(int(d_model))
        self.ln_a_ffn = nn.LayerNorm(int(d_model))
        self.ln_y_ffn = nn.LayerNorm(int(d_model))

    def forward(self, x, a, y, return_attn=False):
        attn_aux = {} if return_attn else None

        if return_attn:
            x_self_out, attn_aux['self_x'] = self.self_x(x, x, x, return_attn=True)
            a_self_out, attn_aux['self_a'] = self.self_a(a, a, a, return_attn=True)
            y_self_out, attn_aux['self_y'] = self.self_y(y, y, y, return_attn=True)
        else:
            x_self_out = self.self_x(x, x, x)
            a_self_out = self.self_a(a, a, a)
            y_self_out = self.self_y(y, y, y)

        x_tilde = self.ln_x_self(x + self.resid_dropout(x_self_out))
        a_tilde = self.ln_a_self(a + self.resid_dropout(a_self_out))
        y_tilde = self.ln_y_self(y + self.resid_dropout(y_self_out))

        if return_attn:
            x_a_out, attn_aux['x_from_a'] = self.x_from_a(x_tilde, a_tilde, a_tilde, return_attn=True)
            x_y_out, attn_aux['x_from_y'] = self.x_from_y(x_tilde, y_tilde, y_tilde, return_attn=True)
            a_x_out, attn_aux['a_from_x'] = self.a_from_x(a_tilde, x_tilde, x_tilde, return_attn=True)
            a_y_out, attn_aux['a_from_y'] = self.a_from_y(a_tilde, y_tilde, y_tilde, return_attn=True)
            y_x_out, attn_aux['y_from_x'] = self.y_from_x(y_tilde, x_tilde, x_tilde, return_attn=True)
            y_a_out, attn_aux['y_from_a'] = self.y_from_a(y_tilde, a_tilde, a_tilde, return_attn=True)
        else:
            x_a_out = self.x_from_a(x_tilde, a_tilde, a_tilde)
            x_y_out = self.x_from_y(x_tilde, y_tilde, y_tilde)
            a_x_out = self.a_from_x(a_tilde, x_tilde, x_tilde)
            a_y_out = self.a_from_y(a_tilde, y_tilde, y_tilde)
            y_x_out = self.y_from_x(y_tilde, x_tilde, x_tilde)
            y_a_out = self.y_from_a(y_tilde, a_tilde, a_tilde)

        x_from_a = self.ln_x_from_a(x_tilde + self.resid_dropout(x_a_out))
        x_from_y = self.ln_x_from_y(x_tilde + self.resid_dropout(x_y_out))
        a_from_x = self.ln_a_from_x(a_tilde + self.resid_dropout(a_x_out))
        a_from_y = self.ln_a_from_y(a_tilde + self.resid_dropout(a_y_out))
        y_from_x = self.ln_y_from_x(y_tilde + self.resid_dropout(y_x_out))
        y_from_a = self.ln_y_from_a(y_tilde + self.resid_dropout(y_a_out))

        x_bar = x_from_a + x_from_y
        a_bar = a_from_x + a_from_y
        y_bar = y_from_x + y_from_a

        x_out = self.ln_x_ffn(x_bar + self.ffn_x(x_bar))
        a_out = self.ln_a_ffn(a_bar + self.ffn_a(a_bar))
        y_out = self.ln_y_ffn(y_bar + self.ffn_y(y_bar))

        if return_attn:
            return x_out, a_out, y_out, attn_aux
        return x_out, a_out, y_out


class CausalTransformerForceModel(nn.Module):
    """3-stream masked causal transformer for residual-force prediction.

    Streams:
        X: covariates/state sequence, shape (B, T, d_x)
        A: previous-treatment/control sequence, shape (B, T, d_a)
        Y: previous-outcome/residual sequence, shape (B, T, d_y)

    Output:
        force_seq_pred: (B, T, 3)
        force_pred: (B, 3), last token
    """

    def __init__(
        self,
        d_x,
        d_a,
        d_y,
        seq_len,
        d_model=64,
        n_heads=4,
        n_blocks=2,
        d_qkv=16,
        ff_dim=128,
        dropout=0.1,
        attn_dropout=0.1,
        lmax=8,
        use_out_proj=False,
    ):
        super().__init__()
        self.d_x = int(d_x)
        self.d_a = int(d_a)
        self.d_y = int(d_y)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_blocks = int(n_blocks)
        self.d_qkv = int(d_qkv)
        self.ff_dim = int(ff_dim)

        self.x_proj = nn.Linear(self.d_x, self.d_model)
        self.a_proj = nn.Linear(self.d_a, self.d_model)
        self.y_proj = nn.Linear(self.d_y, self.d_model)
        self.input_dropout = nn.Dropout(float(dropout))

        self.rel_pos = SharedRelativePositionalEncoding(lmax=lmax, d_qkv=self.d_qkv)

        self.blocks = nn.ModuleList(
            [
                CausalTransformerMultiInputBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_qkv=self.d_qkv,
                    ff_dim=self.ff_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_out_proj=use_out_proj,
                    rel_pos=self.rel_pos,
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.phi_proj = nn.Linear(self.d_model, self.d_model)
        self.head = nn.Linear(self.d_model, 3)

    @staticmethod
    def build_teacher_forcing_input(y_target_seq, y0=None):
        """Build shifted previous-outcome stream for teacher forcing.

        Args:
            y_target_seq: (B, T, 3) target residual sequence.
            y0: Optional initial token (B, 1, 3). Defaults to zeros.

        Returns:
            y_in: (B, T, 3), where y_in[:, 1:] = y_target_seq[:, :-1].
        """
        if y_target_seq.dim() != 3 or y_target_seq.size(-1) != 3:
            raise ValueError("y_target_seq must have shape (B,T,3)")

        bsz, t_len, _ = y_target_seq.shape
        if y0 is None:
            y0 = torch.zeros(bsz, 1, 3, dtype=y_target_seq.dtype, device=y_target_seq.device)
        if y0.shape != (bsz, 1, 3):
            raise ValueError(f"Expected y0 shape {(bsz, 1, 3)}, got {tuple(y0.shape)}")

        y_in = torch.zeros_like(y_target_seq)
        y_in[:, 0:1, :] = y0
        if t_len > 1:
            y_in[:, 1:, :] = y_target_seq[:, :-1, :]
        return y_in

    def _check_shape(self, x_seq, a_seq, y_seq):
        if x_seq.dim() != 3 or a_seq.dim() != 3 or y_seq.dim() != 3:
            raise ValueError("x/a/y must be rank-3 tensors.")
        if x_seq.size(0) != a_seq.size(0) or x_seq.size(0) != y_seq.size(0):
            raise ValueError("x/a/y batch sizes must match.")
        if x_seq.size(1) != a_seq.size(1) or x_seq.size(1) != y_seq.size(1):
            raise ValueError("x/a/y sequence lengths must match.")
        if x_seq.size(2) != self.d_x:
            raise ValueError(f"Expected d_x={self.d_x}, got {x_seq.size(2)}")
        if a_seq.size(2) != self.d_a:
            raise ValueError(f"Expected d_a={self.d_a}, got {a_seq.size(2)}")
        if y_seq.size(2) != self.d_y:
            raise ValueError(f"Expected d_y={self.d_y}, got {y_seq.size(2)}")

    def forward(self, x_seq, a_seq, y_seq, return_attn=False):
        """Forward pass.

        Args:
            x_seq: (B, T, d_x)
            a_seq: (B, T, d_a)
            y_seq: (B, T, d_y)
            return_attn: Return per-block attention weights if True.

        Returns:
            Dict with keys:
                force_seq_pred: (B, T, 3)
                force_pred: (B, 3)
                phi: (B, T, d_model)
                attn(optional): list[dict[str, Tensor]]
        """
        self._check_shape(x_seq, a_seq, y_seq)

        x = self.input_dropout(self.x_proj(x_seq))
        a = self.input_dropout(self.a_proj(a_seq))
        y = self.input_dropout(self.y_proj(y_seq))

        attn_list = []
        for blk in self.blocks:
            if return_attn:
                x, a, y, attn_aux = blk(x, a, y, return_attn=True)
                attn_list.append(attn_aux)
            else:
                x, a, y = blk(x, a, y, return_attn=False)

        phi_tilde = (x + a + y) / 3.0
        phi = F.elu(self.phi_proj(phi_tilde))
        force_seq = self.head(phi)

        out = {
            'force_seq_pred': force_seq,
            'force_pred': force_seq[:, -1, :],
            'phi': phi,
        }
        if return_attn:
            out['attn'] = attn_list
        return out

    def autoregressive_predict(self, x_seq, a_seq, y0=None, return_used_y=False):
        """Autoregressive rollout using previous predicted outputs.

        Args:
            x_seq: (B, T, d_x)
            a_seq: (B, T, d_a)
            y0: Optional initial previous outcome (B, 1, 3).
            return_used_y: Return actually used y-stream input if True.

        Returns:
            force_seq_pred: (B, T, 3)
            used_y(optional): (B, T, 3), with used_y[:, 1:] = pred[:, :-1].
        """
        if x_seq.dim() != 3 or a_seq.dim() != 3:
            raise ValueError("x_seq and a_seq must be rank-3 tensors.")
        if x_seq.size(0) != a_seq.size(0) or x_seq.size(1) != a_seq.size(1):
            raise ValueError("x_seq and a_seq must match in batch and time dims.")

        bsz, t_len, _ = x_seq.shape
        if y0 is None:
            y0 = torch.zeros(bsz, 1, 3, dtype=x_seq.dtype, device=x_seq.device)
        if y0.shape != (bsz, 1, 3):
            raise ValueError(f"Expected y0 shape {(bsz, 1, 3)}, got {tuple(y0.shape)}")

        used_y = torch.zeros(bsz, t_len, 3, dtype=x_seq.dtype, device=x_seq.device)
        used_y[:, 0:1, :] = y0
        preds = torch.zeros(bsz, t_len, 3, dtype=x_seq.dtype, device=x_seq.device)

        for t in range(t_len):
            out = self.forward(x_seq[:, : t + 1, :], a_seq[:, : t + 1, :], used_y[:, : t + 1, :], return_attn=False)
            pred_t = out['force_pred']
            preds[:, t, :] = pred_t
            if t + 1 < t_len:
                used_y[:, t + 1, :] = pred_t

        if return_used_y:
            return preds, used_y
        return preds

    def shared_relative_pe_ok(self):
        """Check whether all attention modules share one relative-PE object."""
        target_id = id(self.rel_pos)
        for blk in self.blocks:
            modules = [
                blk.self_x, blk.self_a, blk.self_y,
                blk.x_from_a, blk.x_from_y,
                blk.a_from_x, blk.a_from_y,
                blk.y_from_x, blk.y_from_a,
            ]
            for mod in modules:
                if id(mod.rel_pos) != target_id:
                    return False
        return True
