"""
Loss functions for training anomalous diffusion predictors.

Following the AnDi Challenge convention (https://www.nature.com/articles/s41467-021-26320-w),
we support:
- Task 1: α (anomalous exponent) prediction - uses MSE loss
- Task 2: Diffusion model classification - uses Cross-Entropy loss

Combined losses for multi-task learning are also provided.

Trajectories are normalized so that MSD(τ) ∝ τ^α.
"""

from typing import Dict, Optional, Tuple, Union
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
        
        batch_size, seq_len, n_dim = trajectory.shape
        device = trajectory.device

        if lengths is None:
            lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device)

        # Global max lag based on the longest sequence in the batch
        max_len = lengths.max().item()
        max_lag_global = max(1, int(max_len * self.max_lag_fraction))

        # Pre-compute lag tensors
        lags = torch.arange(1, max_lag_global + 1, device=device, dtype=torch.float32)
        lags_int = torch.arange(1, max_lag_global + 1, device=device, dtype=torch.long)
        log_lags = torch.log(lags)  # (max_lag_global,)

        # Maximum number of displacement positions (for lag=1)
        max_positions = seq_len - 1

        # Create position indices: (max_positions,)
        positions = torch.arange(max_positions, device=device, dtype=torch.long)

        # End indices for each (lag, position): end_idx = position + lag
        # Shape: (max_lag_global, max_positions)
        end_indices = positions.unsqueeze(0) + lags_int.unsqueeze(1)

        # Valid positions mask: end_idx must be < seq_len
        # Shape: (max_lag_global, max_positions)
        pos_valid_global = end_indices < seq_len

        # Clamp end indices to valid range for safe indexing
        end_indices_clamped = end_indices.clamp(max=seq_len - 1)

        # Gather trajectory values at end and start positions
        # Flatten end_indices for advanced indexing: (max_lag_global * max_positions,)
        end_flat = end_indices_clamped.flatten()

        # trajectory_end: (batch, max_lag_global * max_positions, n_dim)
        trajectory_end = trajectory[:, end_flat, :]
        # Reshape to (batch, max_lag_global, max_positions, n_dim)
        trajectory_end = trajectory_end.view(batch_size, max_lag_global, max_positions, n_dim)

        # trajectory_start: (batch, max_positions, n_dim) -> (batch, 1, max_positions, n_dim)
        trajectory_start = trajectory[:, positions, :].unsqueeze(1)

        # Compute displacements for all lags at once
        # Shape: (batch, max_lag_global, max_positions, n_dim)
        disp = trajectory_end - trajectory_start

        # Per-sample validity: position i with lag τ is valid if i + τ < length
        # i.e., position < length - lag
        # Shape: (batch, max_lag_global, max_positions)
        sample_valid = positions.view(1, 1, -1) < (lengths.view(-1, 1, 1) - lags_int.view(1, -1, 1))

        # Combined validity mask
        # Shape: (batch, max_lag_global, max_positions)
        combined_valid = pos_valid_global.unsqueeze(0) & sample_valid

        # Squared displacements summed over spatial dimensions
        # Shape: (batch, max_lag_global, max_positions)
        sq_disp = (disp ** 2).sum(dim=-1)

        # Apply validity mask
        sq_disp_masked = sq_disp * combined_valid.float()

        # Count valid displacements per (batch, lag)
        # Shape: (batch, max_lag_global)
        valid_counts = combined_valid.float().sum(dim=-1).clamp_min(1)

        # Empirical MSD per sample per lag
        # Shape: (batch, max_lag_global)
        msd_empirical = sq_disp_masked.sum(dim=-1) / valid_counts

        # Lag mask: which lags are valid for each sample (length > lag)
        # Shape: (batch, max_lag_global)
        lag_mask = (lengths.unsqueeze(-1) > lags_int.unsqueeze(0)).float()

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


class ModelClassificationLoss(nn.Module):
    """
    Cross-entropy loss for diffusion model classification (Task 2).
    
    Predicts which of the 5 diffusion models generated the trajectory:
    CTRW, FBM, LW, ATTM, SBM
    """
    
    def __init__(self, num_models: int = 5, label_smoothing: float = 0.0):
        super().__init__()
        self.num_models = num_models
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(
        self,
        model_logits: torch.Tensor,
        model_true: torch.Tensor,
        trajectory: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            model_logits: (batch, num_models) raw logits
            model_true: (batch,) ground truth model indices (0-4)
            trajectory: Not used
            lengths: Not used
        """
        return self.ce_loss(model_logits, model_true.long())


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning: α prediction + model classification.
    
    Total = λ_alpha · MSE(α) + λ_model · CE(model)
    
    Optionally can include physics loss:
    Total = λ_alpha · MSE(α) + λ_model · CE(model) + λ_physics · Physics(α)
    """
    
    def __init__(
        self,
        lambda_alpha: float = 1.0,
        lambda_model: float = 1.0,
        lambda_physics: float = 0.0,
        num_models: int = 5,
        max_lag_fraction: float = 0.25,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            lambda_alpha: Weight for alpha MSE loss
            lambda_model: Weight for model classification loss  
            lambda_physics: Weight for physics-informed loss (0 = disabled)
            num_models: Number of diffusion model classes
            max_lag_fraction: For physics loss
            label_smoothing: Label smoothing for classification
        """
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.lambda_model = lambda_model
        self.lambda_physics = lambda_physics
        
        self.alpha_loss = SupervisedLoss()
        self.model_loss = ModelClassificationLoss(num_models, label_smoothing)
        
        if lambda_physics > 0:
            self.physics_loss = PhysicsLoss(max_lag_fraction)
        else:
            self.physics_loss = None
    
    def forward(
        self,
        alpha_pred: Optional[torch.Tensor],
        alpha_true: Optional[torch.Tensor],
        model_pred: Optional[torch.Tensor],
        model_true: Optional[torch.Tensor],
        trajectory: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.
        
        Args:
            alpha_pred: (batch,) predicted α
            alpha_true: (batch,) ground truth α
            model_pred: (batch, num_models) model logits
            model_true: (batch,) ground truth model indices
            trajectory: (batch, T, 2) for physics loss
            lengths: (batch,) sequence lengths
            
        Returns:
            total_loss: scalar
            breakdown: dict with individual loss components
        """
        total = torch.tensor(0.0, device=self._get_device(alpha_pred, model_pred))
        breakdown = {}
        
        # Alpha loss (Task 1)
        if alpha_pred is not None and alpha_true is not None:
            loss_alpha = self.alpha_loss(alpha_pred, alpha_true)
            total = total + self.lambda_alpha * loss_alpha
            breakdown["alpha_mse"] = loss_alpha.item()
        
        # Model classification loss (Task 2)
        if model_pred is not None and model_true is not None:
            loss_model = self.model_loss(model_pred, model_true)
            total = total + self.lambda_model * loss_model
            breakdown["model_ce"] = loss_model.item()
        
        # Optional physics loss
        if self.physics_loss is not None and alpha_pred is not None and trajectory is not None:
            loss_phys = self.physics_loss(alpha_pred, alpha_true, trajectory, lengths)
            total = total + self.lambda_physics * loss_phys
            breakdown["physics"] = loss_phys.item()
        
        breakdown["total"] = total.item()
        
        return total, breakdown
    
    def _get_device(self, *tensors):
        """Get device from first non-None tensor."""
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")


# Registry for easy access
LOSSES = {
    "supervised": SupervisedLoss,
    "physics": PhysicsLoss,
    "combined": CombinedLoss,
    "classification": ModelClassificationLoss,
    "multitask": MultiTaskLoss,
}


def create_loss(name: str, **kwargs) -> nn.Module:
    """Factory function to create a loss by name."""
    if name not in LOSSES:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSSES.keys())}")
    return LOSSES[name](**kwargs)
