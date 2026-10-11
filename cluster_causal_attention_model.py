import math

import torch
import torch.nn as nn


def _causal_support(seq_len, device):
    """Return causal support mask (lower-triangular) with shape (T, T)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def _safe_l2_normalize(x, dim=-1, eps=1e-9):
    """L2-normalize tensor with epsilon for numerical stability."""
    return x / torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=eps)


class ClusterBiasedCausalAttention(nn.Module):
    """Cluster-biased causal self-attention.

    Input:
      x: (B, T, d_model)

    Output:
      y: (B, T, d_model)
      aux: dict with clustering diagnostics and regularizer terms.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        beta=16.0,
        center_c=4.0,
        lambda_same_cluster=0.5,
        lambda_center_token=0.3,
        lambda_other_cluster=0.2,
        lambda_far=0.2,
        far_radius=1.0,
        sep_margin=None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads

        self.beta = float(beta)
        self.center_c = float(center_c)
        self.lambda_same_cluster = float(lambda_same_cluster)
        self.lambda_center_token = float(lambda_center_token)
        self.lambda_other_cluster = float(lambda_other_cluster)
        self.lambda_far = float(lambda_far)
        self.far_radius = float(far_radius)
        self.sep_margin = sep_margin

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(float(dropout))

    def _select_renyi_like_centers(self, key_norm, delta):
        """Greedy center selection in temporal order.

        Args:
            key_norm: (B, H, T, d_head), detached normalized keys.
            delta: separation threshold.

        Returns:
            center_id: (B, H, T) long cluster id per token.
            center_mask: (B, H, T) bool indicating center tokens.
            center_index: (B, H, T) long, center token index assigned per token.
            center_indices_padded: (B, H, T) long, selected centers in order, padded by -1.
            center_counts: (B, H) long.
        """
        bsz, n_heads, seq_len, _ = key_norm.shape
        device = key_norm.device

        center_id = torch.zeros((bsz, n_heads, seq_len), dtype=torch.long, device=device)
        center_mask = torch.zeros((bsz, n_heads, seq_len), dtype=torch.bool, device=device)
        center_index = torch.zeros((bsz, n_heads, seq_len), dtype=torch.long, device=device)
        center_indices_padded = torch.full((bsz, n_heads, seq_len), -1, dtype=torch.long, device=device)
        center_counts = torch.zeros((bsz, n_heads), dtype=torch.long, device=device)

        for b in range(bsz):
            for h in range(n_heads):
                centers = [0]
                center_mask[b, h, 0] = True
                center_id[b, h, 0] = 0
                center_index[b, h, 0] = 0

                for t in range(1, seq_len):
                    kt = key_norm[b, h, t]
                    center_vec = key_norm[b, h, torch.tensor(centers, device=device)]
                    dists = torch.norm(center_vec - kt.unsqueeze(0), dim=-1)
                    min_dist, min_pos = torch.min(dists, dim=0)

                    if float(min_dist.item()) > float(delta):
                        centers.append(t)
                        cid = len(centers) - 1
                        center_mask[b, h, t] = True
                        center_id[b, h, t] = cid
                        center_index[b, h, t] = t
                    else:
                        cid = int(min_pos.item())
                        center_id[b, h, t] = cid
                        center_index[b, h, t] = centers[cid]

                count = len(centers)
                center_counts[b, h] = count
                center_indices_padded[b, h, :count] = torch.tensor(centers, dtype=torch.long, device=device)

        return center_id, center_mask, center_index, center_indices_padded, center_counts

    def _cluster_bias(self, center_id, center_mask, key_norm, support):
        """Build cluster-aware additive bias on attention logits.

        Returns:
            bias: (B, H, T, T)
            dist_ij: (B, H, T, T)
            masks dict for tests/diagnostics.
        """
        dtype = key_norm.dtype
        support_f = support.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

        cid_i = center_id.unsqueeze(-1)  # (B,H,T,1)
        cid_j = center_id.unsqueeze(-2)  # (B,H,1,T)
        same_cluster = (cid_i == cid_j)
        other_cluster = ~same_cluster

        center_key = center_mask.unsqueeze(-2)  # (B,H,1,T)

        dist_ij = torch.norm(
            key_norm.unsqueeze(3) - key_norm.unsqueeze(2),
            dim=-1,
        )  # (B,H,T,T)
        far_mask = dist_ij > float(self.far_radius)

        bias = torch.zeros_like(dist_ij)
        bias = bias + float(self.lambda_same_cluster) * same_cluster.to(dtype)
        bias = bias + float(self.lambda_center_token) * center_key.to(dtype)
        bias = bias - float(self.lambda_other_cluster) * other_cluster.to(dtype)
        bias = bias - float(self.lambda_far) * far_mask.to(dtype)
        bias = bias * support_f

        masks = {
            'same_cluster': same_cluster & support.unsqueeze(0).unsqueeze(0),
            'other_cluster': other_cluster & support.unsqueeze(0).unsqueeze(0),
            'center_key': center_key & support.unsqueeze(0).unsqueeze(0),
            'far_key': far_mask & support.unsqueeze(0).unsqueeze(0),
        }
        return bias, dist_ij, masks

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B,T,d_model), got {tuple(x.shape)}")

        bsz, seq_len, _ = x.shape
        h = self.n_heads
        dh = self.d_head

        q = self.q_proj(x).view(bsz, seq_len, h, dh).transpose(1, 2)  # (B,H,T,Dh)
        k = self.k_proj(x).view(bsz, seq_len, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, h, dh).transpose(1, 2)

        qn = _safe_l2_normalize(q, dim=-1)
        kn = _safe_l2_normalize(k, dim=-1)

        support = _causal_support(seq_len, x.device)
        delta = float(self.center_c) / math.sqrt(max(float(self.beta), 1e-9))

        center_id, center_mask, center_index, center_indices_padded, center_counts = self._select_renyi_like_centers(
            kn.detach(),
            delta=delta,
        )

        bias_cluster, dist_ij, bias_masks = self._cluster_bias(center_id, center_mask, kn, support)

        scores = float(self.beta) * torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(dh))
        logits = scores + bias_cluster

        neg_large = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        logits = logits.masked_fill((~support).unsqueeze(0).unsqueeze(0), neg_large)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (B,H,T,Dh)
        y = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.out_proj(y)

        assigned_center_key = torch.gather(
            kn,
            dim=2,
            index=center_index.unsqueeze(-1).expand(-1, -1, -1, dh),
        )
        compactness = torch.mean(torch.norm(kn - assigned_center_key, dim=-1))

        sep_terms = []
        sep_margin = float(delta if self.sep_margin is None else self.sep_margin)
        for b in range(bsz):
            for hh in range(h):
                c_count = int(center_counts[b, hh].item())
                if c_count < 2:
                    continue
                idx = center_indices_padded[b, hh, :c_count]
                cvec = kn[b, hh, idx]  # (C,Dh)
                dmat = torch.cdist(cvec, cvec, p=2)
                iu = torch.triu_indices(c_count, c_count, offset=1, device=x.device)
                dvals = dmat[iu[0], iu[1]]
                sep_terms.append(torch.mean(torch.clamp(sep_margin - dvals, min=0.0)))
        if len(sep_terms) > 0:
            separation = torch.stack(sep_terms).mean()
        else:
            separation = torch.zeros((), device=x.device, dtype=x.dtype)

        center_mass = torch.mean(
            torch.sum(attn * center_mask.unsqueeze(-2).to(attn.dtype), dim=-1)
        )

        aux = {
            'attn': attn,
            'key_norm': kn.detach(),
            'center_id': center_id,
            'center_mask': center_mask,
            'center_index': center_index,
            'center_indices': center_indices_padded,
            'center_counts': center_counts,
            'delta': torch.tensor(delta, device=x.device, dtype=x.dtype),
            'dist_ij': dist_ij.detach(),
            'bias_masks': bias_masks,
            'compactness': compactness,
            'separation': separation,
            'center_attn_mass': center_mass,
        }
        return y, aux


class ClusterCausalBlock(nn.Module):
    """Transformer block with cluster-biased causal attention."""

    def __init__(
        self,
        d_model,
        n_heads,
        ff_dim,
        dropout=0.1,
        beta=16.0,
        center_c=4.0,
        lambda_same_cluster=0.5,
        lambda_center_token=0.3,
        lambda_other_cluster=0.2,
        lambda_far=0.2,
        far_radius=1.0,
        sep_margin=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))

        self.attn = ClusterBiasedCausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            beta=beta,
            center_c=center_c,
            lambda_same_cluster=lambda_same_cluster,
            lambda_center_token=lambda_center_token,
            lambda_other_cluster=lambda_other_cluster,
            lambda_far=lambda_far,
            far_radius=far_radius,
            sep_margin=sep_margin,
        )
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ff_dim), int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x):
        y, aux = self.attn(self.norm1(x))
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x, aux


class ClusterCausalForceModel(nn.Module):
    """Cluster-biased causal attention force predictor.

    Input:
      hist: (B, T, F)

    Output:
      force_seq_pred: (B, T, 3)
      force_pred: (B, 3), last token
    """

    def __init__(
        self,
        input_dim=14,
        seq_len=12,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
        beta=16.0,
        center_c=4.0,
        lambda_same_cluster=0.5,
        lambda_center_token=0.3,
        lambda_other_cluster=0.2,
        lambda_far=0.2,
        far_radius=1.0,
        sep_margin=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [
                ClusterCausalBlock(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    ff_dim=int(ff_dim),
                    dropout=dropout,
                    beta=beta,
                    center_c=center_c,
                    lambda_same_cluster=lambda_same_cluster,
                    lambda_center_token=lambda_center_token,
                    lambda_other_cluster=lambda_other_cluster,
                    lambda_far=lambda_far,
                    far_radius=far_radius,
                    sep_margin=sep_margin,
                )
                for _ in range(int(n_layers))
            ]
        )

        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, max(self.d_model, int(ff_dim))),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(self.d_model, int(ff_dim)), 3),
        )

    def forward(self, hist, return_aux=False):
        if hist.dim() != 3:
            raise ValueError(f"Expected hist shape (B,T,F), got {tuple(hist.shape)}")
        if hist.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len {self.seq_len}, got {hist.size(1)}")
        if hist.size(2) != self.input_dim:
            raise ValueError(f"Expected feature_dim {self.input_dim}, got {hist.size(2)}")

        z = self.input_proj(hist) + self.pos_embed[:, :hist.size(1), :]

        attn_aux = []
        compact_terms = []
        sep_terms = []
        center_mass_terms = []
        for blk in self.blocks:
            z, aux = blk(z)
            attn_aux.append(aux)
            compact_terms.append(aux['compactness'])
            sep_terms.append(aux['separation'])
            center_mass_terms.append(aux['center_attn_mass'])

        z = self.norm(z)
        force_seq = self.head(z)

        out = {
            'force_seq_pred': force_seq,
            'force_pred': force_seq[:, -1, :],
        }
        if return_aux:
            out.update(
                {
                    'attn_aux': attn_aux,
                    'compactness': torch.stack(compact_terms).mean(),
                    'separation': torch.stack(sep_terms).mean(),
                    'center_attn_mass': torch.stack(center_mass_terms).mean(),
                }
            )
        return out
