"""CNN-based generalized model for anomalous diffusion parameter inference.

Uses 1D convolutions to extract multi-scale features from trajectory data.
Trained ONCE on a large dataset, then used for inference on new trajectories.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .base import InferenceMethod


class ResidualBlock1D(nn.Module):
    """1D Residual block with skip connection."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out


class CNNEncoder(nn.Module):
    """
    1D CNN encoder with dilated convolutions for multi-scale feature extraction.
    
    Inspired by WaveNet architecture for capturing patterns at different time scales.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        base_channels: int = 32,
        num_blocks: int = 4,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.base_channels = base_channels
        
        # Initial projection from (x, y) coordinates
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        
        # Dilated residual blocks with increasing dilation
        self.blocks = nn.ModuleList([
            ResidualBlock1D(base_channels, kernel_size, dilation=2**i)
            for i in range(num_blocks)
        ])
        
        # Global pooling + final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = base_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trajectory tensor of shape (batch, seq_len, 2)
            
        Returns:
            Encoded features of shape (batch, output_dim)
        """
        # Transpose to (batch, channels, seq_len) for conv1d
        x = x.transpose(1, 2)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Apply dilated residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)
        
        return x


class MultiScaleCNNEncoder(nn.Module):
    """
    Multi-scale CNN that processes trajectory at different resolutions.
    
    Captures both fine-grained local motion and coarse global patterns.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        channels: int = 32,
    ):
        super().__init__()
        
        # Fine scale: small kernels for local patterns
        self.fine_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        
        # Medium scale
        self.medium_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        
        # Coarse scale: large kernels for global patterns
        self.coarse_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        
        # Displacement-based features (like computing increments)
        self.disp_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=2, padding=0),  # Computes differences
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels * 4  # 3 scales + displacement
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Trajectory tensor of shape (batch, seq_len, 2)
            
        Returns:
            Encoded features of shape (batch, output_dim)
        """
        x = x.transpose(1, 2)  # (batch, 2, seq_len)
        
        # Multi-scale features
        fine = self.global_pool(self.fine_conv(x)).squeeze(-1)
        medium = self.global_pool(self.medium_conv(x)).squeeze(-1)
        coarse = self.global_pool(self.coarse_conv(x)).squeeze(-1)
        disp = self.disp_conv(x).squeeze(-1)
        
        # Concatenate all scales
        return torch.cat([fine, medium, coarse, disp], dim=-1)


class CNNPredictor(nn.Module):
    """Full CNN model for predicting α and D0 from trajectories."""
    
    def __init__(
        self,
        base_channels: int = 32,
        num_blocks: int = 4,
        use_multiscale: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_multiscale = use_multiscale
        
        if use_multiscale:
            self.encoder = MultiScaleCNNEncoder(
                input_dim=2,
                channels=base_channels,
            )
        else:
            self.encoder = CNNEncoder(
                input_dim=2,
                base_channels=base_channels,
                num_blocks=num_blocks,
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


class CNNInference(InferenceMethod):
    """
    Wrapper for using a pre-trained CNN model for inference.
    
    Unlike per-trajectory methods, this loads a pre-trained model
    and performs inference without any training.
    """
    
    def __init__(
        self,
        model: Optional[CNNPredictor] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            model: Pre-trained CNNPredictor model
            checkpoint_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        super().__init__(name="CNN Predictor")
        self.device = device
        
        if model is not None:
            self.model = model.to(device)
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            # Initialize with default architecture (for training)
            self.model = CNNPredictor().to(device)
        
        self.model.eval()
    
    def _load_checkpoint(self, path: str) -> CNNPredictor:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        config = checkpoint.get("config", {})
        model = CNNPredictor(
            base_channels=config.get("base_channels", 32),
            num_blocks=config.get("num_blocks", 4),
            use_multiscale=config.get("use_multiscale", True),
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
        ).unsqueeze(0).to(self.device)  # (1, T, 2)
        
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


def create_cnn_model(
    base_channels: int = 32,
    num_blocks: int = 4,
    use_multiscale: bool = True,
    dropout: float = 0.1,
) -> CNNPredictor:
    """Factory function to create a CNN model."""
    return CNNPredictor(
        base_channels=base_channels,
        num_blocks=num_blocks,
        use_multiscale=use_multiscale,
        dropout=dropout,
    )

