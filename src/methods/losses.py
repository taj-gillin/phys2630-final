"""
Loss functions for training anomalous diffusion exponent predictors.

Following the AnDi Challenge convention (https://www.nature.com/articles/s41467-021-26320-w),
we focus on predicting α (anomalous exponent) only. Trajectories are normalized
so that MSD(τ) ∝ τ^α.

All losses share the same interface:
    Input:  predictions (alpha), targets (alpha_true), trajectory
    Output: scalar loss value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedLoss(nn.Module):
    """
    Standard supervised loss on α.
    
    Uses MSE on α prediction.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        alpha_pred: torch.Tensor,
        alpha_true: torch.Tensor,
        trajectory: torch.Tensor = None,  # Not used, but kept for interface consistency
        lengths: torch.Tensor = None,  # Not used
    ) -> torch.Tensor:
        """
        Args:
            alpha_pred: (batch,) predicted α
            alpha_true: (batch,) ground truth α
            trajectory: (batch, T, 2) - not used
        """
        return F.mse_loss(alpha_pred, alpha_true)


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss based on MSD power-law relationship.
    
    Following AnDi conventions, enforces: log(MSD) = α·log(τ) + const
    
    The loss fits α by comparing the slope of log(MSD) vs log(τ) between
    the predicted α and the empirical MSD from the trajectory.
    
    This can be used ALONE or COMBINED with supervised loss.
    """
    
    def __init__(self, max_lag_fraction: float = 0.25):
        super().__init__()
        self.max_lag_fraction = max_lag_fraction
    
    def forward(
        self,
        alpha_pred: torch.Tensor,
        alpha_true: torch.Tensor = None,  # Not used
        trajectory: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute physics loss by comparing predicted vs empirical MSD slope.
        
        For normalized trajectories: MSD(τ) ∝ τ^α
        In log-log space: log(MSD) = α·log(τ) + const
        
        Args:
            alpha_pred: (batch,) predicted α
            trajectory: (batch, T, 2) trajectory (padded)
            lengths: (batch,) true lengths before padding
        """
        if trajectory is None:
            raise ValueError("PhysicsLoss requires trajectory input")
        
        batch_size, seq_len, _ = trajectory.shape
        device = trajectory.device

        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device)

        # Global max lag based on the longest sequence in the batch
        max_len = lengths.max().item()
        max_lag_global = max(1, int(max_len * self.max_lag_fraction))

        # Pre-compute log-lags tensor
        lags = torch.arange(1, max_lag_global + 1, device=device, dtype=torch.float32)
        log_lags = torch.log(lags)  # (max_lag_global,)

        # Collect empirical MSDs and valid masks per lag
        msd_empirical_list = []
        lag_mask_list = []

        for lag in range(1, max_lag_global + 1):
            # Displacements for this lag across the whole batch
            disp = trajectory[:, lag:, :] - trajectory[:, :-lag, :]

            # Valid mask: positions before padding for each sample
            valid = (torch.arange(seq_len - lag, device=device)
                     < (lengths - lag).unsqueeze(1))

            # Avoid division by zero
            valid_counts = valid.sum(dim=1).clamp_min(1)

            # Empirical MSD for this lag, per sample
            msd = ((disp ** 2).sum(dim=-1) * valid).sum(dim=1) / valid_counts
            msd_empirical_list.append(msd)

            # Mask indicating which samples have valid entries for this lag
            lag_mask_list.append((lengths > lag).float())

        msd_empirical = torch.stack(msd_empirical_list, dim=1)  # (batch, max_lag_global)
        lag_mask = torch.stack(lag_mask_list, dim=1)            # (batch, max_lag_global)

        # Compute log-MSD
        log_msd_emp = torch.log(msd_empirical + 1e-8)  # (batch, max_lag_global)

        # Theoretical: log(MSD) = α·log(τ) + const
        # We compute the expected slope (alpha) and compare
        # For each sample, the theoretical log-MSD has slope = alpha_pred
        # We fit: log_msd_theo = alpha_pred * log_lags + intercept
        # 
        # To make this differentiable, we compute the intercept that minimizes
        # the weighted MSE for each sample, then compute the residual loss.
        
        # Compute per-sample intercept: intercept = mean(log_msd - alpha * log_lag)
        # weighted by valid lags
        log_lags_batch = log_lags.unsqueeze(0).expand(batch_size, -1)  # (batch, max_lag_global)
        alpha_batch = alpha_pred.unsqueeze(-1)  # (batch, 1)
        
        # Predicted log-MSD with optimal intercept per sample
        residual = log_msd_emp - alpha_batch * log_lags_batch  # (batch, max_lag_global)
        intercept = (residual * lag_mask).sum(dim=1) / lag_mask.sum(dim=1).clamp_min(1.0)  # (batch,)
        
        log_msd_theo = alpha_batch * log_lags_batch + intercept.unsqueeze(-1)  # (batch, max_lag_global)

        # Loss: MSE between theoretical and empirical log-MSD
        sq_err = (log_msd_theo - log_msd_emp) ** 2
        per_traj_loss = (sq_err * lag_mask).sum(dim=1) / lag_mask.sum(dim=1).clamp_min(1.0)

        physics_loss = per_traj_loss.mean()
        
        return physics_loss


class CombinedLoss(nn.Module):
    """
    Combined supervised + physics loss.
    
    Total = supervised_loss + λ · physics_loss
    
    This is what makes a model "physics-informed" (PINN).
    """
    
    def __init__(
        self,
        lambda_physics: float = 0.1,
        max_lag_fraction: float = 0.25,
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.supervised = SupervisedLoss()
        self.physics = PhysicsLoss(max_lag_fraction)
    
    def forward(
        self,
        alpha_pred: torch.Tensor,
        alpha_true: torch.Tensor,
        trajectory: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            breakdown: dict with individual loss components
        """
        loss_sup = self.supervised(alpha_pred, alpha_true, trajectory, lengths)
        loss_phys = self.physics(alpha_pred, alpha_true, trajectory, lengths)
        
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
