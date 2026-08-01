import math

import torch
import torch.nn as nn


class WeightedCausalMultiheadAttention(nn.Module):
    """WCMHA used by PowerFormer.

    Given X in R^{B x P x d_model}, attention logits are:
      S = QK^T / sqrt(d_head)
      S_tilde = S + M_causal + M_decay
    where M_decay supports:
      - weight_powerlaw:     -alpha * log(lag + 1)
      - similarity_powerlaw: -(lag ** alpha)
    with lag = i - j for j <= i.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        mask_type="weight_powerlaw",
        alpha=0.5,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if mask_type not in ("weight_powerlaw", "similarity_powerlaw"):
            raise ValueError(f"Unsupported mask_type: {mask_type}")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.mask_type = mask_type
        self.alpha = float(alpha)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(float(dropout))

    def _build_causal_mask(self, seq_len, device):
        idx = torch.arange(seq_len, device=device)
        return idx.unsqueeze(0) > idx.unsqueeze(1)

    def _build_decay_mask(self, seq_len, device, dtype):
        idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        lag = idx.unsqueeze(1) - idx.unsqueeze(0)
        lag_nonneg = torch.clamp(lag, min=0.0)

        if self.mask_type == "weight_powerlaw":
            decay = -self.alpha * torch.log(lag_nonneg + 1.0)
        elif self.mask_type == "similarity_powerlaw":
            decay = -(lag_nonneg ** self.alpha)
        else:
            raise ValueError(f"Unsupported mask_type: {self.mask_type}")

        return decay.to(dtype=dtype)

    def forward(self, x, return_attn=False):
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B,P,d_model), got {tuple(x.shape)}")

        bsz, seq_len, _ = x.shape
        h, dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(bsz, seq_len, h, dh).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, h, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)

        causal_mask = self._build_causal_mask(seq_len=seq_len, device=x.device)
        decay_mask = self._build_decay_mask(seq_len=seq_len, device=x.device, dtype=scores.dtype)

        # Softmax in float32 for stability, then cast back to model dtype.
        logits = scores.float() + decay_mask.float().unsqueeze(0).unsqueeze(0)
        logits = logits.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_dropout(attn)

        y = torch.matmul(attn.to(dtype=v.dtype), v)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(y)

        if return_attn:
            return out, attn
        return out


class PowerFormerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
        attn_dropout=0.0,
        mask_type="weight_powerlaw",
        alpha=0.5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = WeightedCausalMultiheadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attn_dropout,
            mask_type=mask_type,
            alpha=alpha,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PowerFormerForceModel(nn.Module):
    """Encoder-only PowerFormer for single-step 3D force compensation."""

    def __init__(
        self,
        input_dim=14,
        seq_len=16,
        patch_len=4,
        patch_stride=2,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        head_dropout=0.1,
        attn_dropout=0.0,
        mask_type="weight_powerlaw",
        alpha=0.5,
    ):
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if patch_len < 1 or patch_stride < 1:
            raise ValueError("patch_len and patch_stride must be >= 1")
        if patch_len > seq_len:
            raise ValueError("patch_len must be <= seq_len")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.d_model = int(d_model)

        self.patch_embed = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.patch_stride,
        )

        self.num_patches = ((self.seq_len - self.patch_len) // self.patch_stride) + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [
                PowerFormerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    mask_type=mask_type,
                    alpha=alpha,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        head_hidden = max(int(d_ff), int(d_model))
        self.head = nn.Sequential(
            nn.Linear(self.num_patches * self.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 3),
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B,T,F), got {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len {self.seq_len}, got {x.size(1)}")
        if x.size(2) != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {x.size(2)}")

        z = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        if z.size(1) != self.num_patches:
            raise ValueError(
                f"Patch count mismatch: expected {self.num_patches}, got {z.size(1)}"
            )
        z = z + self.pos_embed

        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        z_flat = z.reshape(z.size(0), -1)
        return self.head(z_flat)
