"""
Loss functions for training diffusion parameter predictors.

All losses share the same interface:
    Input:  predictions (alpha, D0), targets (alpha_true, D0_true), trajectory
    Output: scalar loss value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedLoss(nn.Module):
    """
    Standard supervised loss on α and D₀.
    
    Uses MSE on α and log(D₀) for scale invariance.
    """
    
    def __init__(self, alpha_weight: float = 1.0, D0_weight: float = 0.1):
        super().__init__()
        self.alpha_weight = alpha_weight
        self.D0_weight = D0_weight
    
    def forward(
        self,
        alpha_pred: torch.Tensor,
        D0_pred: torch.Tensor,
        alpha_true: torch.Tensor,
        D0_true: torch.Tensor,
        trajectory: torch.Tensor = None,  # Not used, but kept for interface consistency
        scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            alpha_pred: (batch,) predicted α
            D0_pred: (batch,) predicted D₀ (in normalized space)
            alpha_true: (batch,) ground truth α
            D0_true: (batch,) ground truth D₀
            trajectory: (batch, T, 2) - not used
            scale: (batch,) normalization scale applied to trajectory
        """
        loss_alpha = F.mse_loss(alpha_pred, alpha_true)
        
        # D0 in log space, accounting for normalization
        if scale is not None:
            D0_true_normalized = D0_true / (scale ** 2)
        else:
            D0_true_normalized = D0_true
        
        loss_D0 = F.mse_loss(
            torch.log(D0_pred + 1e-8),
            torch.log(D0_true_normalized + 1e-8)
        )
        
        return self.alpha_weight * loss_alpha + self.D0_weight * loss_D0


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss based on MSD power-law relationship.
    
    Enforces: MSD(τ) = 4 · D₀ · τ^α
    
    This can be used ALONE or COMBINED with supervised loss.
    """
    
    def __init__(self, max_lag_fraction: float = 0.25):
        super().__init__()
        self.max_lag_fraction = max_lag_fraction
    
    def forward(
        self,
        alpha_pred: torch.Tensor,
        D0_pred: torch.Tensor,
        alpha_true: torch.Tensor = None,  # Not used
        D0_true: torch.Tensor = None,  # Not used
        trajectory: torch.Tensor = None,
        scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute physics loss by comparing predicted vs empirical MSD.
        
        Args:
            alpha_pred: (batch,) predicted α
            D0_pred: (batch,) predicted D₀
            trajectory: (batch, T, 2) normalized trajectory
            scale: (batch,) normalization scale (not used here)
        """
        if trajectory is None:
            raise ValueError("PhysicsLoss requires trajectory input")
        
        batch_size, seq_len, _ = trajectory.shape
        max_lag = max(1, int(seq_len * self.max_lag_fraction))
        device = trajectory.device
        
        physics_losses = []
        
        for i in range(batch_size):
            traj = trajectory[i]  # (seq_len, 2)
            
            # Compute empirical MSD for multiple lags
            lags = torch.arange(1, max_lag + 1, device=device, dtype=torch.float32)
            msd_empirical = []
            
            for lag in range(1, max_lag + 1):
                displacements = traj[lag:] - traj[:-lag]
                msd = (displacements ** 2).sum(dim=-1).mean()
                msd_empirical.append(msd)
            
            msd_empirical = torch.stack(msd_empirical)
            
            # Theoretical MSD from predicted parameters
            msd_theoretical = 4.0 * D0_pred[i] * torch.pow(lags, alpha_pred[i])
            
            # Log-scale MSE for better gradient behavior
            log_msd_emp = torch.log(msd_empirical + 1e-8)
            log_msd_theo = torch.log(msd_theoretical + 1e-8)
            
            physics_loss = F.mse_loss(log_msd_theo, log_msd_emp)
            physics_losses.append(physics_loss)
        
        return torch.stack(physics_losses).mean()


class CombinedLoss(nn.Module):
    """
    Combined supervised + physics loss.
    
    Total = supervised_loss + λ · physics_loss
    
    This is what makes a model "physics-informed" (PINN).
    """
    
    def __init__(
        self,
        lambda_physics: float = 0.1,
        alpha_weight: float = 1.0,
        D0_weight: float = 0.1,
        max_lag_fraction: float = 0.25,
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.supervised = SupervisedLoss(alpha_weight, D0_weight)
        self.physics = PhysicsLoss(max_lag_fraction)
    
    def forward(
        self,
        alpha_pred: torch.Tensor,
        D0_pred: torch.Tensor,
        alpha_true: torch.Tensor,
        D0_true: torch.Tensor,
        trajectory: torch.Tensor,
        scale: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            breakdown: dict with individual loss components
        """
        loss_sup = self.supervised(
            alpha_pred, D0_pred, alpha_true, D0_true, trajectory, scale
        )
        loss_phys = self.physics(
            alpha_pred, D0_pred, alpha_true, D0_true, trajectory, scale
        )
        
        total = loss_sup + self.lambda_physics * loss_phys
        
        breakdown = {
            "supervised": loss_sup.item(),
            "physics": loss_phys.item(),
            "total": total.item(),
        }
        
        return total, breakdown


# Registry for easy access
LOSSES = {
    "supervised": SupervisedLoss,
    "physics": PhysicsLoss,
    "combined": CombinedLoss,
}


def create_loss(name: str, **kwargs) -> nn.Module:
    """Factory function to create a loss by name."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSSES.keys())}")
    return LOSSES[name](**kwargs)

