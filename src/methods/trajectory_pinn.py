"""Physics-Informed Neural Network for anomalous diffusion parameter inference.

This is a proper PINN that:
1. Takes trajectory as input → outputs α and D0
2. Trained on a large dataset with physics-informed loss
3. Enforces MSD power-law relationship during training

Unlike per-trajectory training, this learns across all trajectories.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import InferenceMethod


class TrajectoryEncoder(nn.Module):
    """
    Hybrid encoder combining LSTM for temporal patterns and CNN for local features.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        cnn_channels: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # CNN branch for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        
        # LSTM branch for sequential dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim + cnn_channels,  # Original + CNN features
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trajectory tensor of shape (batch, seq_len, 2)
            
        Returns:
            Encoded features of shape (batch, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # CNN features
        x_cnn = x.transpose(1, 2)  # (batch, 2, seq_len)
        cnn_features = self.cnn(x_cnn).transpose(1, 2)  # (batch, seq_len, cnn_channels)
        
        # Concatenate with original trajectory
        x_combined = torch.cat([x, cnn_features], dim=-1)
        
        # LSTM encoding
        _, (h_n, _) = self.lstm(x_combined)
        
        # Concatenate forward and backward final states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        
        return torch.cat([h_forward, h_backward], dim=-1)


class TrajectoryPINN(nn.Module):
    """
    Physics-Informed Neural Network for anomalous diffusion.
    
    Architecture:
        Trajectory → Encoder → (α, D0)
        
    Physics constraint (applied during training):
        MSD(τ) = 4 · D0 · τ^α
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        cnn_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = TrajectoryEncoder(
            input_dim=2,
            hidden_dim=hidden_dim,
            cnn_channels=cnn_channels,
        )
        
        encoder_dim = self.encoder.output_dim
        
        # Shared feature refinement
        self.shared_fc = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Separate prediction heads
        self.alpha_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.D0_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory: (batch, seq_len, 2) tensor of positions
            
        Returns:
            alpha: (batch,) predicted anomalous exponent in [0.1, 2.0]
            D0: (batch,) predicted diffusion coefficient (positive)
        """
        # Encode trajectory
        features = self.encoder(trajectory)
        shared = self.shared_fc(features)
        
        # Predict parameters
        alpha_raw = self.alpha_head(shared).squeeze(-1)
        alpha = torch.sigmoid(alpha_raw) * 1.9 + 0.1  # [0.1, 2.0]
        
        log_D0 = self.D0_head(shared).squeeze(-1)
        D0 = torch.exp(log_D0)
        
        return alpha, D0
    
    def compute_physics_loss(
        self,
        trajectory: torch.Tensor,
        alpha: torch.Tensor,
        D0: torch.Tensor,
        max_lag_fraction: float = 0.25,
    ) -> torch.Tensor:
        """
        Compute physics-informed loss based on MSD power-law.
        
        For each trajectory, computes empirical MSD and compares
        to theoretical prediction: MSD(τ) = 4 · D0 · τ^α
        
        Args:
            trajectory: (batch, seq_len, 2) tensor
            alpha: (batch,) predicted α values
            D0: (batch,) predicted D0 values
            max_lag_fraction: Maximum lag as fraction of sequence length
            
        Returns:
            Physics loss (scalar)
        """
        batch_size, seq_len, _ = trajectory.shape
        max_lag = max(1, int(seq_len * max_lag_fraction))
        
        physics_losses = []
        
        for i in range(batch_size):
            traj = trajectory[i]  # (seq_len, 2)
            
            # Compute empirical MSD for multiple lags
            lags = torch.arange(1, max_lag + 1, device=trajectory.device, dtype=torch.float32)
            msd_empirical = []
            
            for lag in range(1, max_lag + 1):
                displacements = traj[lag:] - traj[:-lag]  # (seq_len - lag, 2)
                msd = (displacements ** 2).sum(dim=-1).mean()
                msd_empirical.append(msd)
            
            msd_empirical = torch.stack(msd_empirical)
            
            # Theoretical MSD from predicted parameters
            msd_theoretical = 4.0 * D0[i] * torch.pow(lags, alpha[i])
            
            # Log-scale loss for better gradient behavior
            log_msd_emp = torch.log(msd_empirical + 1e-8)
            log_msd_theo = torch.log(msd_theoretical + 1e-8)
            
            physics_loss = F.mse_loss(log_msd_theo, log_msd_emp)
            physics_losses.append(physics_loss)
        
        return torch.stack(physics_losses).mean()


class TrajectoryPINNInference(InferenceMethod):
    """
    Wrapper for using a pre-trained Trajectory PINN for inference.
    """
    
    def __init__(
        self,
        model: Optional[TrajectoryPINN] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            model: Pre-trained TrajectoryPINN model
            checkpoint_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        super().__init__(name="Trajectory PINN")
        self.device = device
        
        if model is not None:
            self.model = model.to(device)
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            self.model = TrajectoryPINN().to(device)
        
        self.model.eval()
    
    def _load_checkpoint(self, path: str) -> TrajectoryPINN:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        config = checkpoint.get("config", {})
        model = TrajectoryPINN(
            hidden_dim=config.get("hidden_dim", 64),
            cnn_channels=config.get("cnn_channels", 32),
            dropout=config.get("dropout", 0.1),
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)
    
    def fit(self, trajectory: np.ndarray) -> None:
        """
        Run inference on a trajectory.
        
        Note: This doesn't train anything - it uses the pre-trained model.
        """
        self.reset()
        
        # Convert to tensor
        traj_tensor = torch.tensor(
            trajectory, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Normalize trajectory
        traj_tensor = self._normalize_trajectory(traj_tensor)
        
        # Forward pass
        with torch.no_grad():
            alpha, D0 = self.model(traj_tensor)
        
        self._alpha = alpha.item()
        self._D0 = D0.item()
        self._is_fitted = True
    
    def _normalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Normalize trajectory for consistent input scale."""
        # Center at origin
        traj = traj - traj[:, 0:1, :]
        
        # Scale by standard deviation
        std = traj.std() + 1e-8
        traj = traj / std
        
        return traj


def create_pinn_model(
    hidden_dim: int = 64,
    cnn_channels: int = 32,
    dropout: float = 0.1,
) -> TrajectoryPINN:
    """Factory function to create a Trajectory PINN model."""
    return TrajectoryPINN(
        hidden_dim=hidden_dim,
        cnn_channels=cnn_channels,
        dropout=dropout,
    )

