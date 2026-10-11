import torch
import torch.nn as nn


def build_causal_mask(seq_len, device):
    """Build causal attention mask.

    Args:
        seq_len: Sequence length T.
        device: Torch device.

    Returns:
        Bool tensor of shape (T, T), where True means masked (j > i).
    """
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class WaveletActivation(nn.Module):
    """Learnable wavelet activation: w1 * sin(x) + w2 * cos(x).

    Args:
        hidden_dim: Feature dimension on last axis.

    Input/Output shape:
        x: (..., hidden_dim) -> y: (..., hidden_dim)
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.w1 = nn.Parameter(torch.ones(self.hidden_dim))
        self.w2 = nn.Parameter(torch.zeros(self.hidden_dim))

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class SpatioTemporalMixer(nn.Module):
    """Token mixer mapping per-step input features to model embeddings.

    Input shape:
        x_seq: (B, T, F_in)
    Output shape:
        z: (B, T, d_model)
    """

    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(int(input_dim), int(d_model))
        self.act1 = WaveletActivation(int(d_model))
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(d_model), int(d_model))
        self.act2 = WaveletActivation(int(d_model))
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, x_seq):
        z = self.fc1(x_seq)
        z = self.act1(z)
        z = self.drop1(z)
        z = self.fc2(z)
        z = self.act2(z)
        z = self.drop2(z)
        return z


class PINNsFormerEncoderLayer(nn.Module):
    """Encoder layer with causal self-attention + wavelet FFN."""

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, use_layernorm=False):
        super().__init__()
        self.use_layernorm = bool(use_layernorm)
        self.norm1 = nn.LayerNorm(int(d_model)) if self.use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(int(d_model)) if self.use_layernorm else nn.Identity()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(float(dropout))

        self.ffn_fc1 = nn.Linear(int(d_model), int(ff_dim))
        self.ffn_act = WaveletActivation(int(ff_dim))
        self.ffn_drop = nn.Dropout(float(dropout))
        self.ffn_fc2 = nn.Linear(int(ff_dim), int(d_model))
        self.ffn_out_drop = nn.Dropout(float(dropout))

    def forward(self, x, causal_mask):
        x_n = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_n,
            x_n,
            x_n,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + self.dropout_attn(attn_out)

        y = self.norm2(x)
        y = self.ffn_fc1(y)
        y = self.ffn_act(y)
        y = self.ffn_drop(y)
        y = self.ffn_fc2(y)
        y = self.ffn_out_drop(y)
        x = x + y
        return x


class PINNsFormerDecoderLayer(nn.Module):
    """Decoder layer with cross-attention only (no decoder self-attention)."""

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, use_layernorm=False):
        super().__init__()
        self.use_layernorm = bool(use_layernorm)
        self.norm_q = nn.LayerNorm(int(d_model)) if self.use_layernorm else nn.Identity()
        self.norm_mem = nn.LayerNorm(int(d_model)) if self.use_layernorm else nn.Identity()
        self.norm_ffn = nn.LayerNorm(int(d_model)) if self.use_layernorm else nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(float(dropout))

        self.ffn_fc1 = nn.Linear(int(d_model), int(ff_dim))
        self.ffn_act = WaveletActivation(int(ff_dim))
        self.ffn_drop = nn.Dropout(float(dropout))
        self.ffn_fc2 = nn.Linear(int(ff_dim), int(d_model))
        self.ffn_out_drop = nn.Dropout(float(dropout))

    def forward(self, dec, memory, causal_mask):
        q = self.norm_q(dec)
        m = self.norm_mem(memory)
        attn_out, _ = self.cross_attn(
            q,
            m,
            m,
            attn_mask=causal_mask,
            need_weights=False,
        )
        dec = dec + self.dropout_attn(attn_out)

        y = self.norm_ffn(dec)
        y = self.ffn_fc1(y)
        y = self.ffn_act(y)
        y = self.ffn_drop(y)
        y = self.ffn_fc2(y)
        y = self.ffn_out_drop(y)
        dec = dec + y
        return dec


class PINNsFormerDynamicsModel(nn.Module):
    """PINNsFormer-inspired causal sequence model for quadrotor dynamics.

    Architecture:
      1) Causal history sequence X_seq in R^{B x T x F}
      2) Spatio-Temporal Mixer -> token embeddings in R^{B x T x d_model}
      3) Encoder with causal self-attention
      4) Decoder with cross-attention only (no decoder self-attention)
      5) Output head -> dynamics sequence in R^{B x T x D_out}

    Notes:
      - Decoder input shares same token embeddings as encoder input.
      - Current-step output uses last token y[:, -1, :].
    """

    def __init__(
        self,
        input_dim=14,
        seq_len=5,
        d_model=32,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ff_dim=128,
        dropout=0.1,
        out_dim=6,
        use_layernorm=False,
        use_time_feature=True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.out_dim = int(out_dim)
        self.use_time_feature = bool(use_time_feature)

        mixer_in_dim = self.input_dim + (1 if self.use_time_feature else 0)
        self.mixer = SpatioTemporalMixer(
            input_dim=mixer_in_dim,
            d_model=self.d_model,
            dropout=dropout,
        )

        self.encoder_layers = nn.ModuleList(
            [
                PINNsFormerEncoderLayer(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    ff_dim=int(ff_dim),
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                )
                for _ in range(int(n_encoder_layers))
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                PINNsFormerDecoderLayer(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    ff_dim=int(ff_dim),
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                )
                for _ in range(int(n_decoder_layers))
            ]
        )

        self.head_fc1 = nn.Linear(self.d_model, int(ff_dim))
        self.head_act = WaveletActivation(int(ff_dim))
        self.head_drop = nn.Dropout(float(dropout))
        self.head_fc2 = nn.Linear(int(ff_dim), self.out_dim)

    def _append_time_feature(self, x_seq):
        # x_seq: (B, T, F)
        bsz, seq_len, _ = x_seq.shape
        if not self.use_time_feature:
            return x_seq
        tau = torch.linspace(0.0, 1.0, steps=seq_len, device=x_seq.device, dtype=x_seq.dtype)
        tau = tau.view(1, seq_len, 1).expand(bsz, -1, -1)
        return torch.cat((x_seq, tau), dim=-1)

    def forward(self, x_seq):
        """Forward pass.

        Args:
            x_seq: Tensor of shape (B, T, F).

        Returns:
            Dict:
              dyn_seq_pred: (B, T, D_out)
              dyn_pred: (B, D_out)  # last token
        """
        if x_seq.dim() != 3:
            raise ValueError(f"Expected input shape (B,T,F), got {tuple(x_seq.shape)}")
        if x_seq.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_seq.size(1)}")
        if x_seq.size(2) != self.input_dim:
            raise ValueError(f"Expected feature_dim={self.input_dim}, got {x_seq.size(2)}")

        x_aug = self._append_time_feature(x_seq)  # (B, T, F or F+1)
        tokens = self.mixer(x_aug)                # (B, T, d_model)

        causal_mask = build_causal_mask(self.seq_len, x_seq.device)

        memory = tokens
        for layer in self.encoder_layers:
            memory = layer(memory, causal_mask=causal_mask)

        dec = tokens
        for layer in self.decoder_layers:
            dec = layer(dec, memory=memory, causal_mask=causal_mask)

        y = self.head_fc1(dec)
        y = self.head_act(y)
        y = self.head_drop(y)
        y = self.head_fc2(y)

        if not torch.all(torch.isfinite(y)):
            raise ValueError("Non-finite PINNsFormer output detected.")

        return {
            'dyn_seq_pred': y,
            'dyn_pred': y[:, -1, :],
        }
