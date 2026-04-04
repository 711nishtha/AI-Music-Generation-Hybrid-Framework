"""
Mel-Spectrogram Diffusion Renderer (Audio Rendering Layer)
===========================================================
Architecture:
  - U-Net backbone operating on mel spectrograms (80 mel x 512 time)
  - Time-step conditioning via sinusoidal embedding + MLP
  - Cross-attention conditioning on symbolic token embeddings
  - DDPM training with linear / cosine noise schedule
  - At inference: DDIM 50-step sampling for fast generation
  - ~25M parameters in small config
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t.unsqueeze(1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int, d_time: int = 256):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(d_time)
        self.mlp = nn.Sequential(
            nn.Linear(d_time, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sin_emb(t))


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, d_time: int,
                 groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(),
                                       nn.Linear(d_time, out_ch * 2))
        self.norm2  = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.drop   = nn.Dropout(dropout)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip   = (nn.Conv2d(in_ch, out_ch, 1)
                       if in_ch != out_ch else nn.Identity())

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock2D(nn.Module):
    """Spatial self-attention + cross-attention to symbolic conditioning."""

    def __init__(self, channels: int, n_heads: int = 4,
                 d_context: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm_self   = nn.GroupNorm(min(8, channels), channels)
        self.norm_cross  = nn.GroupNorm(min(8, channels), channels)
        self.self_attn   = nn.MultiheadAttention(channels, n_heads,
                                                  batch_first=True,
                                                  dropout=dropout)
        self.cross_attn  = nn.MultiheadAttention(channels, n_heads,
                                                   kdim=d_context, vdim=d_context,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: Tensor,
                context: Optional[Tensor] = None) -> Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)

        x_n = self.norm_self(x).view(B, C, H * W).permute(0, 2, 1)
        sa, _ = self.self_attn(x_n, x_n, x_n)
        x_flat = x_flat + sa

        if context is not None:
            x_n = self.norm_cross(x_flat.permute(0, 2, 1).view(B, C, H, W))
            x_n = x_n.view(B, C, H * W).permute(0, 2, 1)
            ca, _ = self.cross_attn(x_n, context, context)
            x_flat = x_flat + ca

        x_flat = x_flat + self.ff(x_flat)
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, d_time: int,
                 d_context: int, attn: bool = False,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.res1 = ResnetBlock(in_ch,  out_ch, d_time, dropout=dropout)
        self.res2 = ResnetBlock(out_ch, out_ch, d_time, dropout=dropout)
        self.attn = AttentionBlock2D(out_ch, n_heads, d_context, dropout) \
                    if attn else None

    def forward(self, x: Tensor, t_emb: Tensor,
                context: Optional[Tensor] = None) -> Tensor:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, context)
        return x


class MelDiffusionRenderer(nn.Module):
    """
    Conditional DDPM on mel spectrograms.
    Input:  noisy mel (B, 1, 80, 512) + time-step scalar + symbolic context
    Output: noise prediction (B, 1, 80, 512)
    """

    SMALL_CONFIG = dict(
        d_model=128, channel_mults=(1, 2, 2, 4),
        attn_resolutions=(False, True, True, True),
        n_heads=4, d_context=256, T_steps=1000, dropout=0.1
    )
    FULL_CONFIG = dict(
        d_model=256, channel_mults=(1, 2, 4, 8),
        attn_resolutions=(False, False, True, True),
        n_heads=8, d_context=512, T_steps=1000, dropout=0.1
    )

    def __init__(
        self,
        vocab_size:   int,
        d_model:      int   = 128,
        channel_mults: Tuple = (1, 2, 2, 4),
        attn_resolutions: Tuple = (False, True, True, True),
        n_heads:      int   = 4,
        d_context:    int   = 256,
        T_steps:      int   = 1000,
        dropout:      float = 0.1,
        n_mel:        int   = 80,
        max_sym_len:  int   = 1024,
    ):
        super().__init__()
        self.T_steps   = T_steps
        self.n_mel     = n_mel

        self.sym_embedding = nn.Embedding(vocab_size, d_context, padding_idx=0)
        self.sym_encoder   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_context, n_heads, d_context * 2,
                                        dropout=dropout, batch_first=True,
                                        norm_first=True),
            num_layers=4
        )

        self.time_emb = TimeEmbedding(d_model, d_time=128)

        base_ch = d_model
        chs     = [base_ch * m for m in channel_mults]

        self.input_conv = nn.Conv2d(1, base_ch, 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downsample  = nn.ModuleList()
        in_ch = base_ch
        for i, (ch, do_attn) in enumerate(zip(chs, attn_resolutions)):
            self.enc_blocks.append(UNetBlock(
                in_ch, ch, d_model, d_context, do_attn, n_heads, dropout))
            in_ch = ch
            if i < len(chs) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.downsample.append(nn.Identity())

        self.mid_block1 = ResnetBlock(in_ch, in_ch, d_model, dropout=dropout)
        self.mid_attn   = AttentionBlock2D(in_ch, n_heads, d_context, dropout)
        self.mid_block2 = ResnetBlock(in_ch, in_ch, d_model, dropout=dropout)

        self.dec_blocks  = nn.ModuleList()
        self.upsample    = nn.ModuleList()
        rev_chs = list(reversed(chs))
        rev_attn = list(reversed(attn_resolutions))
        for i, (ch, do_attn) in enumerate(zip(rev_chs, rev_attn)):
            skip_ch = rev_chs[i]
            self.dec_blocks.append(UNetBlock(
                in_ch + skip_ch, ch, d_model, d_context, do_attn, n_heads, dropout))
            in_ch = ch
            if i < len(rev_chs) - 1:
                self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.upsample.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(8, in_ch), in_ch)
        self.out_conv = nn.Conv2d(in_ch, 1, 3, padding=1)

        betas = torch.linspace(1e-4, 0.02, T_steps)
        alphas          = 1.0 - betas
        alphas_cumprod  = alphas.cumprod(0)
        self.register_buffer("betas",          betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",
                             alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1.0 - alphas_cumprod).sqrt())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode_symbolic(self, sym_ids: Tensor,
                          sym_mask: Optional[Tensor]) -> Tensor:
        emb = self.sym_embedding(sym_ids)
        key_pad = (sym_mask == 0) if sym_mask is not None else None
        ctx = self.sym_encoder(emb, src_key_padding_mask=key_pad)
        return ctx

    def forward(
        self,
        x:        Tensor,
        t:        Tensor,
        sym_ids:  Tensor,
        sym_mask: Optional[Tensor] = None,
    ) -> Tensor:
        ctx     = self._encode_symbolic(sym_ids, sym_mask)
        t_emb   = self.time_emb(t)

        h = self.input_conv(x)

        skip_h = []
        for blk, ds in zip(self.enc_blocks, self.downsample):
            h = blk(h, t_emb, ctx)
            skip_h.append(h)
            h = ds(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, ctx)
        h = self.mid_block2(h, t_emb)

        for blk, us in zip(self.dec_blocks, self.upsample):
            skip = skip_h.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = blk(h, t_emb, ctx)
            h = us(h)

        return self.out_conv(F.silu(self.out_norm(h)))

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sb = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sa * x0 + sb * noise, noise

    def training_loss(self, x0: Tensor, sym_ids: Tensor,
                      sym_mask: Optional[Tensor] = None) -> Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T_steps, (B,), device=x0.device)
        xt, noise = self.q_sample(x0, t)
        pred_noise = self(xt, t, sym_ids, sym_mask)
        return F.mse_loss(pred_noise, noise)

    @torch.inference_mode()
    def ddim_sample(
        self,
        sym_ids:  Tensor,
        sym_mask: Optional[Tensor] = None,
        n_steps:  int   = 50,
        eta:      float = 0.0,
        shape:    Optional[Tuple] = None,
    ) -> Tensor:
        self.eval()
        B = sym_ids.shape[0]
        if shape is None:
            shape = (B, 1, self.n_mel, 512)

        device = sym_ids.device
        x = torch.randn(shape, device=device)

        step_size = self.T_steps // n_steps
        timesteps = list(range(0, self.T_steps, step_size))[::-1]

        acp = self.alphas_cumprod

        for i, t_val in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)

            eps   = self(x, t_tensor, sym_ids, sym_mask)
            ac_t  = acp[t_val]
            ac_tp = acp[t_prev] if t_prev > 0 else torch.tensor(1.0)

            x0_pred = (x - (1 - ac_t).sqrt() * eps) / ac_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            sigma   = eta * ((1 - ac_tp) / (1 - ac_t)).sqrt() * \
                      ((1 - ac_t / ac_tp)).sqrt()
            dir_xt  = (1 - ac_tp - sigma ** 2).sqrt() * eps
            noise   = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = ac_tp.sqrt() * x0_pred + dir_xt + noise

        return x

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
