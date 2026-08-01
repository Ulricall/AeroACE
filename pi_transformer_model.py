import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_causal_support(seq_len, device):
    """Return lower-triangular causal support mask of shape (T, T)."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def hilbert_imag(signal):
    """Compute Hilbert transform imaginary part for real signal.

    Args:
        signal: Real tensor with shape (..., T).

    Returns:
        Tensor with shape (..., T) representing imag(Hilbert(signal)).
    """
    if signal.dim() < 1:
        raise ValueError("signal must have at least 1 dimension")

    n = signal.size(-1)
    x_fft = torch.fft.fft(signal, dim=-1)
    h = torch.zeros(n, device=signal.device, dtype=signal.dtype)

    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0

    analytic = torch.fft.ifft(x_fft * h.to(dtype=x_fft.dtype), dim=-1)
    return analytic.imag


def symmetric_kl_divergence(series_attn, prior_attn, stopgrad_series=False, stopgrad_prior=False, eps=1e-9):
    """Compute symmetric KL over causal support.

    Args:
        series_attn: Tensor of shape (..., T, T), row-normalized.
        prior_attn: Tensor of shape (..., T, T), row-normalized.
        stopgrad_series: Detach series branch in divergence term.
        stopgrad_prior: Detach prior branch in divergence term.
        eps: Numerical epsilon.

    Returns:
        Scalar tensor.
    """
    if series_attn.shape != prior_attn.shape:
        raise ValueError(
            f"series/prior shape mismatch: {tuple(series_attn.shape)} vs {tuple(prior_attn.shape)}"
        )

    s = series_attn.detach() if stopgrad_series else series_attn
    p = prior_attn.detach() if stopgrad_prior else prior_attn

    s = torch.clamp(s, min=eps)
    p = torch.clamp(p, min=eps)

    kl_sp = s * (torch.log(s) - torch.log(p))
    kl_ps = p * (torch.log(p) - torch.log(s))

    seq_len = series_attn.size(-1)
    support = _build_causal_support(seq_len, series_attn.device)
    while support.dim() < series_attn.dim():
        support = support.unsqueeze(0)
    support = support.to(dtype=series_attn.dtype)

    delta = (kl_sp + kl_ps) * support
    denom = torch.clamp(support.expand_as(delta).sum(), min=1.0)
    return delta.sum() / denom


def prior_regularization_terms(H_list, tau_list, tau_ref=1.0):
    """Compute prior regularization terms from per-layer H/tau lists.

    Args:
        H_list: list of tensors, each shape (B, n_heads, T).
        tau_list: list of tensors, each shape (B, n_heads, T).
        tau_ref: reference tau value.

    Returns:
        Dict with scalar tensors: {'smooth', 'prior', 'distill'}.
    """
    if len(H_list) == 0 or len(tau_list) == 0:
        raise ValueError("H_list and tau_list must be non-empty")

    smooth_terms = []
    prior_terms = []
    for H, tau in zip(H_list, tau_list):
        if tau.size(-1) > 1:
            smooth_terms.append(torch.mean((tau[:, :, 1:] - tau[:, :, :-1]) ** 2))
        else:
            smooth_terms.append(torch.zeros((), device=tau.device, dtype=tau.dtype))
        prior_terms.append(torch.mean(H ** 2) + torch.mean((tau - tau_ref) ** 2))

    smooth = torch.stack(smooth_terms).mean()
    prior = torch.stack(prior_terms).mean()
    distill = torch.zeros((), device=smooth.device, dtype=smooth.dtype)
    return {'smooth': smooth, 'prior': prior, 'distill': distill}


class PiTransformerBlock(nn.Module):
    """Pi-Transformer-inspired dual-attention block.

    Input/Output shape: x in R^{B x T x d_model}.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
        gamma=1.0,
        sigma=4.0,
        tau_eps=1e-3,
        alpha_prior=0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = self.d_model // self.n_heads
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.tau_eps = float(tau_eps)
        self.alpha_prior = float(alpha_prior)

        self.norm_attn = nn.LayerNorm(self.d_model)
        self.norm_ffn = nn.LayerNorm(self.d_model)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.h_head = nn.Linear(self.d_model, self.n_heads)
        self.tau_head = nn.Linear(self.d_model, self.n_heads)
        self.phase_head = nn.Linear(self.d_model, self.n_heads)

        self.dropout = nn.Dropout(float(dropout))
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, int(d_ff)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), self.d_model),
            nn.Dropout(float(dropout)),
        )

    def _series_attention(self, x_norm):
        bsz, seq_len, _ = x_norm.shape
        q = self.q_proj(x_norm).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        support = _build_causal_support(seq_len, x_norm.device)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill((~support).unsqueeze(0).unsqueeze(0), neg_inf)
        series_attn = torch.softmax(scores, dim=-1)
        series_context = torch.matmul(series_attn, v)
        return series_attn, series_context, v

    def _prior_attention(self, x_norm, eps):
        bsz, seq_len, _ = x_norm.shape
        support = _build_causal_support(seq_len, x_norm.device)
        support_f = support.to(dtype=x_norm.dtype)

        H = torch.tanh(self.h_head(x_norm)).transpose(1, 2)  # (B, Hn, T)
        tau = F.softplus(self.tau_head(x_norm)).transpose(1, 2) + self.tau_eps

        psi = torch.cumsum(torch.exp(self.gamma * H), dim=-1)  # (B, Hn, T)
        dpsi = psi.unsqueeze(-1) - psi.unsqueeze(-2)  # (B, Hn, T, T)
        w_time = torch.exp(-(dpsi ** 2) / (2.0 * (self.sigma ** 2)))

        u = self.phase_head(x_norm).transpose(1, 2)  # (B, Hn, T)
        hilbert_u = hilbert_imag(u)
        theta = torch.atan2(hilbert_u, u)

        dtheta = theta.unsqueeze(-1) - theta.unsqueeze(-2)  # (B, Hn, T, T)
        phase_num = torch.sin(0.5 * dtheta) ** 2
        phase_den = torch.clamp(2.0 * (tau.unsqueeze(-1) ** 2), min=eps)
        w_phase = torch.exp(-phase_num / phase_den)

        A = w_time * w_phase * support_f.unsqueeze(0).unsqueeze(0)
        P = A / torch.clamp(A.sum(dim=-1, keepdim=True), min=eps)
        return P, H, tau, theta

    def forward(self, x, eps=1e-9):
        """Forward pass.

        Args:
            x: Tensor with shape (B, T, d_model).
            eps: Numerical epsilon.

        Returns:
            x_out: Tensor with shape (B, T, d_model).
            aux: dict with S/P/H/tau/theta tensors.
        """
        x_norm = self.norm_attn(x)

        series_attn, series_context, v = self._series_attention(x_norm)
        prior_attn, H, tau, theta = self._prior_attention(x_norm, eps=eps)

        if self.alpha_prior != 0.0:
            prior_context = torch.matmul(prior_attn, v)
            merged_context = series_context + self.alpha_prior * prior_context
        else:
            merged_context = series_context

        bsz, _, seq_len, _ = merged_context.shape
        merged_context = merged_context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

        x = x + self.dropout(self.out_proj(merged_context))
        x = x + self.ffn(self.norm_ffn(x))

        aux = {
            'series_attn': series_attn,
            'prior_attn': prior_attn,
            'H': H,
            'tau': tau,
            'theta': theta,
        }
        return x, aux


class PiTransformerForceModel(nn.Module):
    """Prior-informed dual-attention force predictor.

    Args:
        x: input history tensor in R^{B x T x F}.

    Returns:
        f_hat: force prediction tensor in R^{B x 3}.
        aux (optional): diagnostics dict.
    """

    def __init__(
        self,
        input_dim=14,
        seq_len=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        gamma=1.0,
        sigma=4.0,
        tau_eps=1e-3,
        alpha_prior=0.0,
    ):
        super().__init__()
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)

        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [
                PiTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    gamma=gamma,
                    sigma=sigma,
                    tau_eps=tau_eps,
                    alpha_prior=alpha_prior,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, max(self.d_model, int(d_ff))),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(self.d_model, int(d_ff)), 3),
        )

    def forward(self, x, return_aux=False, eps=1e-9):
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (B,T,F), got {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len {self.seq_len}, got {x.size(1)}")
        if x.size(2) != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {x.size(2)}")

        z = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]

        series_attn_list = []
        prior_attn_list = []
        H_list = []
        tau_list = []
        theta_list = []

        for blk in self.blocks:
            z, aux = blk(z, eps=eps)
            series_attn_list.append(aux['series_attn'])
            prior_attn_list.append(aux['prior_attn'])
            H_list.append(aux['H'])
            tau_list.append(aux['tau'])
            theta_list.append(aux['theta'])

        z = self.norm(z)
        f_hat = self.head(z[:, -1, :])

        if not return_aux:
            return f_hat

        mismatch = torch.stack(
            [torch.mean(torch.abs(s - p)) for s, p in zip(series_attn_list, prior_attn_list)]
        ).mean()

        return f_hat, {
            'series_attn': series_attn_list,
            'prior_attn': prior_attn_list,
            'H': H_list,
            'tau': tau_list,
            'theta': theta_list,
            'mean_mismatch': mismatch,
        }
