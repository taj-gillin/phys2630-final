"""LSTM-based generalized model for anomalous diffusion parameter inference.

This model is trained ONCE on a large dataset of trajectories and can then
predict α for any new trajectory without retraining.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .base import InferenceMethod


class LSTMEncoder(nn.Module):
    """LSTM encoder for trajectory sequences."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Output dimension accounts for bidirectional
        self.output_dim = hidden_dim * self.num_directions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trajectory tensor of shape (batch, seq_len, 2)
            
        Returns:
            Encoded features of shape (batch, output_dim)
        """
        # LSTM forward pass
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        # Take the last layer's hidden states
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2]  # Last layer, forward
            h_backward = h_n[-1]  # Last layer, backward
            out = torch.cat([h_forward, h_backward], dim=-1)
        else:
            out = h_n[-1]
        
        return out


class LSTMPredictor(nn.Module):
    """Full LSTM model for predicting α and D0 from trajectories."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.encoder = LSTMEncoder(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        
        encoder_dim = self.encoder.output_dim
        
        # Prediction heads
        self.alpha_head = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.D0_head = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        features = self.encoder(trajectory)
        
        # Alpha prediction: sigmoid to [0, 1], then scale to [0.1, 2.0]
        alpha_raw = self.alpha_head(features).squeeze(-1)
        alpha = torch.sigmoid(alpha_raw) * 1.9 + 0.1
        
        # D0 prediction: exp to ensure positivity
        log_D0 = self.D0_head(features).squeeze(-1)
        D0 = torch.exp(log_D0)
        
        return alpha, D0


class LSTMInference(InferenceMethod):
    """
    Wrapper for using a pre-trained LSTM model for inference.
    
    Unlike per-trajectory methods, this loads a pre-trained model
    and performs inference without any training.
    """
    
    def __init__(
        self,
        model: Optional[LSTMPredictor] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            model: Pre-trained LSTMPredictor model
            checkpoint_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        super().__init__(name="LSTM Predictor")
        self.device = device
        
        if model is not None:
            self.model = model.to(device)
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            # Initialize with default architecture (for training)
            self.model = LSTMPredictor().to(device)
        
        self.model.eval()
    
    def _load_checkpoint(self, path: str) -> LSTMPredictor:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Extract model config if saved
        config = checkpoint.get("config", {})
        model = LSTMPredictor(
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            bidirectional=config.get("bidirectional", True),
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
        ).unsqueeze(0).to(self.device)  # (1, T, 2)
        
        # Normalize trajectory (center and scale)
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


def create_lstm_model(
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    bidirectional: bool = True,
) -> LSTMPredictor:
    """Factory function to create an LSTM model."""
    return LSTMPredictor(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )

