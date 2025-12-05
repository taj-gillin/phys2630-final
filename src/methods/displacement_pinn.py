"""Displacement distribution PINN for anomalous diffusion parameter inference."""

import numpy as np
import torch
import torch.nn as nn

from .base import InferenceMethod
from .networks import MLP


class DisplacementPINN(InferenceMethod):
    """
    Physics-informed neural network for displacement distribution modeling.
    
    The PINN learns the probability density P(Δx, Δy | τ) of displacements
    at different lag times, with physics constraint from the diffusion equation.
    
    Physics:
        For anomalous diffusion, the displacement distribution evolves as:
        ∂P/∂τ = D(τ) · (∂²P/∂Δx² + ∂²P/∂Δy²)
        where D(τ) = D0 · α · τ^(α-1)
        
    The solution is Gaussian with variance σ²(τ) = 2·D0·τ^α for each dimension.
    """
    
    def __init__(
        self,
        hidden_layers: int = 4,
        hidden_dim: int = 32,
        epochs: int = 1000,
        lr: float = 0.001,
        lambda_physics: float = 0.1,
        alpha_init: float = 0.8,
        D0_init: float = 1.0,
        max_lag_fraction: float = 0.25,
        n_collocation: int = 500,
        verbose: bool = False,
    ):
        """
        Args:
            hidden_layers: Number of hidden layers in MLP
            hidden_dim: Width of hidden layers
            epochs: Number of training epochs
            lr: Learning rate
            lambda_physics: Weight for physics loss
            alpha_init: Initial guess for alpha
            D0_init: Initial guess for D0
            max_lag_fraction: Maximum lag as fraction of trajectory length
            n_collocation: Number of collocation points for physics loss
            verbose: Print progress during training
        """
        super().__init__(name="Displacement PINN")
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.lambda_physics = lambda_physics
        self.alpha_init = alpha_init
        self.D0_init = D0_init
        self.max_lag_fraction = max_lag_fraction
        self.n_collocation = n_collocation
        self.verbose = verbose
        self._loss_history: list[dict] = []
    
    def fit(self, trajectory: np.ndarray) -> None:
        """
        Fit the displacement distribution PINN to trajectory.
        
        Args:
            trajectory: Array of shape (T, 2) with (x, y) positions
        """
        self.reset()
        self._loss_history = []
        
        # Extract displacements at multiple lag times
        displacements, lags = self._extract_displacements(trajectory)
        
        if len(displacements) < 10:
            self._alpha = np.nan
            self._D0 = np.nan
            self._is_fitted = True
            return
        
        # Normalize displacements for numerical stability
        disp_std = np.std(displacements[:, :2])
        if disp_std < 1e-8:
            disp_std = 1.0
        
        # Convert to tensors
        dx = torch.tensor(displacements[:, 0] / disp_std, dtype=torch.float32)
        dy = torch.tensor(displacements[:, 1] / disp_std, dtype=torch.float32)
        tau = torch.tensor(displacements[:, 2], dtype=torch.float32)
        
        # Build network: inputs (Δx, Δy, τ) → output log P
        net = MLP(
            in_dim=3,
            out_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.hidden_layers,
        )
        
        # Learnable physics parameters
        alpha = nn.Parameter(torch.tensor(self.alpha_init))
        log_D0 = nn.Parameter(torch.tensor(np.log(self.D0_init)))
        
        optimizer = torch.optim.Adam(
            list(net.parameters()) + [alpha, log_D0],
            lr=self.lr,
        )
        
        # Collocation points for physics loss
        tau_max = float(tau.max())
        tau_min = float(tau.min())
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # === Data Loss: Gaussian likelihood ===
            # For anomalous diffusion, displacement variance is 2*D0*τ^α per dimension
            D0 = torch.exp(log_D0)
            variance = 2.0 * D0 * torch.pow(tau, alpha)
            
            # Gaussian log-likelihood (normalized displacement space)
            log_prob = -0.5 * (dx**2 + dy**2) / (variance / disp_std**2) - torch.log(variance / disp_std**2) - np.log(2 * np.pi)
            loss_data = -log_prob.mean()
            
            # === Physics Loss: Check that network output matches Gaussian ===
            # Sample collocation points
            tau_coll = torch.rand(self.n_collocation) * (tau_max - tau_min) + tau_min
            dx_coll = torch.randn(self.n_collocation) * 2
            dy_coll = torch.randn(self.n_collocation) * 2
            
            # Network prediction at collocation points
            inp_coll = torch.stack([dx_coll, dy_coll, tau_coll], dim=-1)
            log_P_net = net(inp_coll).squeeze()
            
            # Theoretical Gaussian log-probability
            var_coll = 2.0 * D0 * torch.pow(tau_coll, alpha)
            log_P_theory = -0.5 * (dx_coll**2 + dy_coll**2) / var_coll - torch.log(var_coll) - np.log(2 * np.pi)
            
            # Physics loss: network should match Gaussian
            loss_physics = torch.mean((log_P_net - log_P_theory) ** 2)
            
            # Total loss
            loss = loss_data + self.lambda_physics * loss_physics
            
            loss.backward()
            optimizer.step()
            
            # Clamp alpha to valid range
            with torch.no_grad():
                alpha.clamp_(0.1, 2.0)
            
            self._loss_history.append({
                "total": loss.item(),
                "data": loss_data.item(),
                "physics": loss_physics.item(),
            })
            
            if self.verbose and epoch % 200 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.4f}, α={alpha.item():.4f}, D0={D0.item():.4f}")
        
        # Account for normalization in D0
        self._alpha = alpha.item()
        self._D0 = torch.exp(log_D0).item() * disp_std**2
        self._is_fitted = True
    
    def _extract_displacements(self, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract displacement samples at multiple lag times.
        
        Returns:
            displacements: Array of shape (N, 3) with columns (Δx, Δy, τ)
            unique_lags: Array of unique lag times used
        """
        T = len(trajectory)
        max_lag = max(1, int(T * self.max_lag_fraction))
        
        samples = []
        lags_used = []
        
        for lag in range(1, max_lag + 1):
            for t in range(T - lag):
                dx = trajectory[t + lag, 0] - trajectory[t, 0]
                dy = trajectory[t + lag, 1] - trajectory[t, 1]
                samples.append([dx, dy, lag])
            lags_used.append(lag)
        
        return np.array(samples), np.array(lags_used)
    
    @property
    def loss_history(self) -> list[dict]:
        """Training loss history with breakdown."""
        return self._loss_history


