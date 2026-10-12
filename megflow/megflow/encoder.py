"""Raw temporal-attention encoder matching the prototype's operations."""

import math

import torch
from torch import nn


class RawEncoder(nn.Module):
    def __init__(self, n_sensors=102, widths=(64, 128, 128), kernel_size=15,
                 n_queries=4, cond_dim=192):
        super().__init__()
        if len(widths) != 3 or kernel_size % 2 != 1:
            raise ValueError('raw encoder needs three widths and an odd kernel')
        channels = [n_sensors, *widths]
        self.convs = nn.ModuleList(nn.Conv1d(a, b, kernel_size, padding=kernel_size // 2)
                                   for a, b in zip(channels[:-1], channels[1:]))
        self.norms = nn.ModuleList(nn.LayerNorm(c, eps=1e-5) for c in widths)
        self.key = nn.Linear(widths[-1], widths[-1])
        self.value = nn.Linear(widths[-1], widths[-1])
        self.queries = nn.Parameter(torch.randn(n_queries, widths[-1]) / math.sqrt(widths[-1]))
        self.projection = nn.Linear(n_queries * widths[-1], cond_dim)

    def forward(self, y):
        # Input is already mean-pooled by 12, without sensor standardisation.
        for conv, norm in zip(self.convs, self.norms):
            y = torch.relu(norm(conv(y).transpose(1, 2))).transpose(1, 2)
        tokens = y.transpose(1, 2)
        logits = self.queries @ self.key(tokens).transpose(1, 2) / math.sqrt(tokens.shape[-1])
        pooled = logits.softmax(dim=-1) @ self.value(tokens)
        return self.projection(pooled.flatten(1)).unsqueeze(1)
