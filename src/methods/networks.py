"""Neural network building blocks for PINN methods."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()
        
        acts = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        act = acts.get(activation, nn.Tanh())
        
        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers + [out_dim]
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


