"""
Hierarchical Music Transformer (Symbolic Planning Layer)
=========================================================
Architecture:
  - Two-level hierarchy:
      * Structure Encoder: processes structure / phrase / chord tokens
        at coarser resolution (1 token per bar)
      * Detail Decoder:    generates full note-level token sequence,
        cross-attending to structure encoder output
  - Rotary positional embeddings (RoPE) for long-range coherence
  - Grouped-query attention for efficiency
  - ~15M parameters in small config, ~80M in full config
"""

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: Tensor, xk: Tensor,
                     freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    xq_    = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_    = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.shape[1]].unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 is_causal: bool = True, cross_attn: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.head_dim  = d_model // n_heads
        self.is_causal = is_causal
        self.cross_attn = cross_attn
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                freqs_cis: Optional[Tensor] = None) -> Tensor:
        B, T, C = x.shape
        kv_src   = context if self.cross_attn and context is not None else x
        S        = kv_src.shape[1]

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None and not self.cross_attn:
            q2, k2 = apply_rotary_emb(
                q.transpose(1, 2), k.transpose(1, 2), freqs_cis
            )
            q = q2.transpose(1, 2)
            k = k2.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.is_causal and not self.cross_attn:
            causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        attn   = self.dropout(F.softmax(attn, dim=-1))
        out    = torch.matmul(attn, v)
        out    = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, cross_attn: bool = False):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout,
                                             is_causal=True, cross_attn=False)
        self.cross_attn_layer = None
        if cross_attn:
            self.norm_cross = nn.LayerNorm(d_model)
            self.cross_attn_layer = MultiHeadAttention(
                d_model, n_heads, dropout, is_causal=False, cross_attn=True)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: Tensor, context: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                freqs_cis: Optional[Tensor] = None) -> Tensor:
        x = x + self.self_attn(self.norm1(x), mask=mask, freqs_cis=freqs_cis)
        if self.cross_attn_layer is not None and context is not None:
            x = x + self.cross_attn_layer(self.norm_cross(x), context=context)
        x = x + self.ff(self.norm2(x))
        return x


class StructureEncoder(nn.Module):
    """Encodes a compressed bar-level sequence to provide structural context."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.layers    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, cross_attn=False)
            for _ in range(n_layers)
        ])
        self.norm  = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_len)

    def forward(self, ids: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        B, T = ids.shape
        pos  = torch.arange(T, device=ids.device).unsqueeze(0)
        x    = self.drop(self.embedding(ids) + self.pos_emb(pos))
        fc   = self.freqs_cis.to(ids.device)
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=fc)
        return self.norm(x)


class HierarchicalMusicTransformer(nn.Module):
    """
    Full hierarchical transformer for music generation.

    Forward pass:
      input_ids  (B, T)  -> token ids for teacher-forced training
      Returns logits (B, T, vocab_size)

    Generation:
      generate(prompt_ids, max_new_tokens) -> (B, T+max_new) token ids
    """

    SMALL_CONFIG = dict(
        vocab_size=None,
        d_model=256,
        n_heads=8,
        n_layers_enc=4,
        n_layers_dec=6,
        d_ff=1024,
        max_len=2048,
        dropout=0.1,
    )
    FULL_CONFIG = dict(
        vocab_size=None,
        d_model=512,
        n_heads=16,
        n_layers_enc=6,
        n_layers_dec=12,
        d_ff=2048,
        max_len=4096,
        dropout=0.1,
    )

    def __init__(
        self,
        vocab_size: int,
        d_model:    int = 256,
        n_heads:    int = 8,
        n_layers_enc: int = 4,
        n_layers_dec: int = 6,
        d_ff:       int = 1024,
        max_len:    int = 2048,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.max_len    = max_len

        self.structure_encoder = StructureEncoder(
            vocab_size=vocab_size, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers_enc,
            d_ff=d_ff, max_len=max_len // 4, dropout=dropout
        )

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.dec_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout,
                             cross_attn=(i % 2 == 0))
            for i in range(n_layers_dec)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, vocab_size, bias=False)
        self.drop     = nn.Dropout(dropout)

        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_len)

        self.head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _extract_structure_tokens(self, ids: Tensor) -> Tensor:
        from src.data.tokenizer import TOKEN2ID
        from src.data.tokenizer import (
            STRUCTURE_TOKENS, PHRASE_TOKENS, CHORD_TOKENS, NO_CHORD_TOKEN,
            BAR_TOKEN, TEMPO_TOKENS, STYLE_TOKENS, EMOTION_TOKENS
        )
        keep_ids = set()
        for t in (STRUCTURE_TOKENS + PHRASE_TOKENS + CHORD_TOKENS +
                  NO_CHORD_TOKEN + BAR_TOKEN + TEMPO_TOKENS +
                  STYLE_TOKENS + EMOTION_TOKENS):
            if t in TOKEN2ID:
                keep_ids.add(TOKEN2ID[t])

        B, T = ids.shape
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for kid in keep_ids:
            mask |= (ids == kid)
        struct_seqs = []
        max_struct_len = max(1, T // 8)
        for b in range(B):
            kept = ids[b][mask[b]][:max_struct_len]
            pad  = torch.full((max_struct_len - len(kept),), 0,
                              device=ids.device, dtype=ids.dtype)
            struct_seqs.append(torch.cat([kept, pad]))
        return torch.stack(struct_seqs)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)

        struct_ids  = self._extract_structure_tokens(input_ids)
        struct_ctx  = self.structure_encoder(struct_ids)

        x = self.drop(self.embedding(input_ids) + self.pos_emb(pos))
        fc = self.freqs_cis[:T].to(input_ids.device)
        for layer in self.dec_layers:
            x = layer(x, context=struct_ctx,
                      mask=attention_mask, freqs_cis=fc)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=0,
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 1024,
        temperature: float  = 0.95,
        top_k: int          = 50,
        top_p: float        = 0.92,
        repetition_penalty: float = 1.1,
    ) -> Tensor:
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.max_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for tid in generated[0].unique():
                    logits[0, tid] /= repetition_penalty

            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0
            sorted_probs /= sorted_probs.sum()

            next_tok = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            generated = torch.cat([generated, next_tok], dim=1)

            if next_tok.item() == 1:
                break

        return generated

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
