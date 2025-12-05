"""MSD-constrained PINN for anomalous diffusion parameter inference."""

import numpy as np
import torch
import torch.nn as nn

from .base import InferenceMethod


class MSDPINN(InferenceMethod):
    """
    Physics-informed approach using MSD power-law constraint.
    
    Learns alpha and D0 by fitting the physics formula:
        MSD(τ) = 4 * D0 * τ^α
    
    This is essentially curve fitting with autograd optimization,
    but framed as a PINN for consistency.
    """
    
    def __init__(
        self,
        epochs: int = 500,
        lr: float = 0.01,
        alpha_init: float = 0.8,
        D0_init: float = 1.0,
        max_lag_fraction: float = 0.25,
        verbose: bool = False,
    ):
        """
        Args:
            epochs: Number of optimization steps
            lr: Learning rate
            alpha_init: Initial guess for alpha
            D0_init: Initial guess for D0
            max_lag_fraction: Maximum lag as fraction of trajectory length
            verbose: Print progress during training
        """
        super().__init__(name="MSD PINN")
        self.epochs = epochs
        self.lr = lr
        self.alpha_init = alpha_init
        self.D0_init = D0_init
        self.max_lag_fraction = max_lag_fraction
        self.verbose = verbose
        self._loss_history: list[float] = []
    
    def fit(self, trajectory: np.ndarray) -> None:
        """
        Fit alpha and D0 using gradient descent on physics loss.
        
        Args:
            trajectory: Array of shape (T, 2) with (x, y) positions
        """
        self.reset()
        self._loss_history = []
        
        # Compute empirical MSD
        lags, msd_empirical = self._compute_msd(trajectory)
        
        # Filter valid points
        mask = (lags > 0) & (msd_empirical > 0)
        if np.sum(mask) < 3:
            self._alpha = np.nan
            self._D0 = np.nan
            self._is_fitted = True
            return
        
        lags = lags[mask]
        msd_empirical = msd_empirical[mask]
        
        # Convert to tensors
        tau = torch.tensor(lags, dtype=torch.float32)
        msd_target = torch.tensor(msd_empirical, dtype=torch.float32)
        
        # Learnable parameters (use log for D0 to ensure positivity)
        alpha = nn.Parameter(torch.tensor(self.alpha_init))
        log_D0 = nn.Parameter(torch.tensor(np.log(self.D0_init)))
        
        # Use separate learning rates - alpha needs smaller lr for stability
        optimizer = torch.optim.Adam([
            {'params': [alpha], 'lr': self.lr * 0.5},
            {'params': [log_D0], 'lr': self.lr},
        ])
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Physics-based MSD prediction
            D0 = torch.exp(log_D0)
            msd_pred = 4.0 * D0 * torch.pow(tau, alpha)
            
            # Loss: relative error on linear scale (more stable gradients for alpha)
            loss = torch.mean(((msd_pred - msd_target) / msd_target) ** 2)
            
            loss.backward()
            
            # Clip gradients to prevent instability
            torch.nn.utils.clip_grad_norm_([alpha, log_D0], max_norm=1.0)
            
            optimizer.step()
            
            # Keep alpha in valid range
            with torch.no_grad():
                alpha.clamp_(0.1, 2.0)
            
            self._loss_history.append(loss.item())
            
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.6f}, α={alpha.item():.4f}, D0={torch.exp(log_D0).item():.4f}")
        
        self._alpha = alpha.item()
        self._D0 = torch.exp(log_D0).item()
        self._is_fitted = True
    
    def _compute_msd(self, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean squared displacement for multiple lag times."""
        T = len(trajectory)
        max_lag = max(1, int(T * self.max_lag_fraction))
        lags = np.arange(1, max_lag + 1)
        msd = np.zeros(len(lags))
        
        for i, lag in enumerate(lags):
            disps = trajectory[lag:] - trajectory[:-lag]
            msd[i] = np.mean(np.sum(disps ** 2, axis=-1))
        
        return lags, msd
    
    @property
    def loss_history(self) -> list[float]:
        """Training loss history."""
        return self._loss_history

