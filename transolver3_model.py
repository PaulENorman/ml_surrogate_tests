"""Adapter for the official Transolver-3 model, installed at a pinned commit."""

from __future__ import annotations

import torch


class SurfaceTransolver3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from models.Transolver_chunk_opt_matrix_mul import Model

        # Match the existing Transolver++ experiment's capacity and outputs.
        # These are intentionally smaller than the paper's 24-layer benchmark.
        self.core = Model(space_dim=6, fun_dim=0, out_dim=5,
                          n_layers=4, n_hidden=128, n_head=4,
                          slice_num=32, mlp_ratio=1, unified_pos=False)

    def forward(self, normals: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        features = torch.cat((embedding, normals), dim=-1)
        return self.core([features], use_checkpoint=False)[0]
