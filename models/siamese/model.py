from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class SiameseConfig:
    embedding_dim: int = 16
    hidden_dim: int = 64
    dropout: float = 0.1


class ProductEncoder(nn.Module):
    def __init__(self, cardinalities: Dict[str, int], config: SiameseConfig) -> None:
        super().__init__()

        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(num_embeddings=size, embedding_dim=config.embedding_dim)
                for name, size in cardinalities.items()
            }
        )
        total_embed = len(cardinalities) * config.embedding_dim + 1  # +1 for price_usd

        self.mlp = nn.Sequential(
            nn.Linear(total_embed, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x_cat: Dict[str, torch.Tensor], x_price: torch.Tensor) -> torch.Tensor:
        embedded = [self.embeddings[name](x_cat[name]) for name in self.embeddings.keys()]
        dense_input = torch.cat(embedded + [x_price.unsqueeze(-1)], dim=-1)
        return self.mlp(dense_input)


class SiamesePricingNetwork(nn.Module):
    def __init__(self, cardinalities: Dict[str, int], config: SiameseConfig) -> None:
        super().__init__()
        self.encoder = ProductEncoder(cardinalities=cardinalities, config=config)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        left_cat: Dict[str, torch.Tensor],
        left_price: torch.Tensor,
        right_cat: Dict[str, torch.Tensor],
        right_price: torch.Tensor,
    ) -> torch.Tensor:
        left_repr = self.encoder(left_cat, left_price)
        right_repr = self.encoder(right_cat, right_price)
        diff = left_repr - right_repr
        return self.head(diff).squeeze(-1)
