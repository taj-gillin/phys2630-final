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
        lengths: torch.Tensor = None,  # Not used
    ) -> torch.Tensor:
        """
        Args:
            alpha_pred: (batch,) predicted α
            D0_pred: (batch,) predicted D₀
            alpha_true: (batch,) ground truth α
            D0_true: (batch,) ground truth D₀
            trajectory: (batch, T, 2) - not used
        """
        loss_alpha = F.mse_loss(alpha_pred, alpha_true)
        
        loss_D0 = F.mse_loss(
            torch.log(D0_pred + 1e-8),
            torch.log(D0_true + 1e-8)
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
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute physics loss by comparing predicted vs empirical MSD.
        
        Args:
            alpha_pred: (batch,) predicted α
            D0_pred: (batch,) predicted D₀
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

        # Pre-compute theoretical lags tensor once
        lags = torch.arange(1, max_lag_global + 1, device=device, dtype=torch.float32)  # (max_lag_global,)

        # Collect empirical MSDs and valid masks per lag
        msd_empirical_list = []
        lag_mask_list = []

        for lag in range(1, max_lag_global + 1):
            # Displacements for this lag across the whole batch (padded trajectories)
            disp = trajectory[:, lag:, :] - trajectory[:, :-lag, :]  # (batch, seq_len - lag, 2)

            # Valid mask: positions before padding for each sample
            valid = (torch.arange(seq_len - lag, device=device)
                     < (lengths - lag).unsqueeze(1))  # (batch, seq_len - lag)

            # Avoid division by zero
            valid_counts = valid.sum(dim=1).clamp_min(1)  # (batch,)

            # Empirical MSD for this lag, per sample
            msd = ( (disp ** 2).sum(dim=-1) * valid ).sum(dim=1) / valid_counts  # (batch,)
            msd_empirical_list.append(msd)

            # Mask indicating which samples have any valid entries for this lag
            lag_mask_list.append((lengths > lag).float())

        msd_empirical = torch.stack(msd_empirical_list, dim=1)  # (batch, max_lag_global)
        lag_mask = torch.stack(lag_mask_list, dim=1)            # (batch, max_lag_global)

        # Theoretical MSD for all lags, per sample
        msd_theoretical = 4.0 * D0_pred.unsqueeze(-1) * torch.pow(lags.unsqueeze(0), alpha_pred.unsqueeze(-1))

        # Log-scale MSE with masking; normalize per sample by its valid lags
        log_msd_emp = torch.log(msd_empirical + 1e-8)
        log_msd_theo = torch.log(msd_theoretical + 1e-8)

        sq_err = (log_msd_theo - log_msd_emp) ** 2  # (batch, max_lag_global)
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
        lengths: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            breakdown: dict with individual loss components
        """
        loss_sup = self.supervised(
            alpha_pred, D0_pred, alpha_true, D0_true, trajectory, lengths
        )
        loss_phys = self.physics(
            alpha_pred, D0_pred, alpha_true, D0_true, trajectory, lengths
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

