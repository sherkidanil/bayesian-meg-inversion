"""Fixed-K baseline and a source-permutation-equivariant AdaLN velocity field."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('frequencies', torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2)))
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):
        phase = t[:, None] * self.frequencies[None] * 1000
        return self.mlp(torch.cat([phase.cos(), phase.sin()], dim=-1))


class AdaLN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, time):
        shift, scale = self.modulation(time).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    def __init__(self, width, heads, context_dim):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(context_dim, 2 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x, context, valid_keys=None):
        batch, tokens, width = x.shape
        q = self.q(x).reshape(batch, tokens, self.heads, width // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        v = v.reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        # SDPA boolean True means a key is allowed, unlike MultiheadAttention masks.
        mask = None if valid_keys is None else valid_keys[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return self.out(y.transpose(1, 2).reshape(batch, tokens, width))


class SetBlock(nn.Module):
    def __init__(self, width, heads, d_cond):
        super().__init__()
        self.norms = nn.ModuleList(AdaLN(width) for _ in range(3))
        self.self_attn = Attention(width, heads, width)
        self.cross_attn = Attention(width, heads, d_cond)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, h, time, cond, mask):
        z = self.norms[0](h, time)
        h = h + self.self_attn(z, z, mask)
        h = h + self.cross_attn(self.norms[1](h, time), cond)
        h = h + self.mlp(self.norms[2](h, time))
        return h.masked_fill(~mask[..., None], 0)


class SetVelocityField(nn.Module):
    def __init__(self, d_in=3, d_model=256, n_layers=6, n_heads=8, d_cond=192):
        super().__init__()
        if d_model % n_heads or d_model % 2:
            raise ValueError('model width must be even and divisible by number of heads')
        self.input = nn.Linear(d_in, d_model)
        self.time = TimeEmbedding(d_model)
        self.blocks = nn.ModuleList(SetBlock(d_model, n_heads, d_cond) for _ in range(n_layers))
        self.norm = AdaLN(d_model)
        self.output = nn.Linear(d_model, d_in)

    def forward(self, t, theta_t, cond, mask):
        h = self.input(theta_t).masked_fill(~mask[..., None], 0)
        time = self.time(t)
        for block in self.blocks:
            h = block(h, time, cond, mask)
        return self.output(self.norm(h, time)).masked_fill(~mask[..., None], 0)


class MLPVelocityField(nn.Module):
    def __init__(self, d_cond=192, width=512, depth=3):
        super().__init__()
        layers = [nn.Linear(9 + d_cond + 1, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.SiLU()])
        layers.append(nn.Linear(width, 9))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t, theta_t, cond, mask):
        if theta_t.shape[1:] != (3, 3):
            raise ValueError('prototype MLP only supports K=3 and xyz')
        inputs = torch.cat([theta_t.flatten(1), cond.mean(dim=1), t[:, None]], dim=-1)
        return self.mlp(inputs).reshape(-1, 3, 3).masked_fill(~mask[..., None], 0)


class ConditionalFlow(nn.Module):
    def __init__(self, encoder, field):
        super().__init__()
        self.encoder, self.field = encoder, field

    def forward(self, t, theta_t, y, mask):
        return self.field(t, theta_t, self.encoder(y), mask)
