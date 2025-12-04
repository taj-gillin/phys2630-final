import torch
import torch.nn as nn
from typing import Tuple
from .networks import MLP


class AnomalousDiffusionPINN(nn.Module):
    def __init__(self, hidden_layers: int = 6, hidden_dim: int = 64, activation: str = "tanh", alpha_init: float = 0.9, D0_init: float = 1.0):
        super().__init__()
        self.net = MLP(in_dim=3, hidden_dim=hidden_dim, num_layers=hidden_layers, activation=activation)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.D0 = nn.Parameter(torch.tensor(float(D0_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def physics_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,3) tensor with requires_grad=True
        returns residual of PDE: u_t - D(t)*(u_xx + u_yy)
        """
        x = x.requires_grad_(True)
        u = self.forward(x)

        grads = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_t = grads[:, 2:3]

        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]

        t = x[:, 2:3]
        D_t = self.D0 * torch.pow(t + 1e-6, self.alpha - 1)
        residual = u_t - D_t * (u_xx + u_yy)
        return residual

    def learned_parameters(self) -> Tuple[float, float]:
        return float(self.alpha.detach().cpu()), float(self.D0.detach().cpu())

