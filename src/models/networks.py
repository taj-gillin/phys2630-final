import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, activation: str = "tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}
        act = acts.get(activation, nn.Tanh())

        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers + [1]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

