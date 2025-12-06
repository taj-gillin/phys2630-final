"""
Unified Diffusion Predictor: combines any encoder with prediction head.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w),
this model predicts only α (anomalous diffusion exponent).

This is the main model class that:
1. Takes a trajectory as input
2. Uses an encoder to extract features
3. Predicts α

The encoder and loss are modular - mix and match as needed.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from .base import InferenceMethod
from .encoders import create_encoder, ENCODERS
from .losses import create_loss, LOSSES


class DiffusionPredictor(nn.Module):
    """
    Universal model for predicting α from trajectories.
    
    Architecture:
        trajectory → encoder → features → head → α
    
    The encoder is pluggable (Linear, MLP, LSTM, CNN, Hybrid).
    """
    
    def __init__(
        self,
        encoder_name: str = "lstm",
        encoder_kwargs: dict = None,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            encoder_name: One of 'linear', 'mlp', 'lstm', 'cnn', 'hybrid'
            encoder_kwargs: Arguments passed to encoder constructor
            head_hidden_dim: Hidden dimension for prediction head
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder_name = encoder_name
        encoder_kwargs = encoder_kwargs or {}
        
        # Create encoder
        self.encoder = create_encoder(encoder_name, **encoder_kwargs)
        encoder_dim = self.encoder.output_dim
        
        # Shared feature refinement
        self.shared_fc = nn.Sequential(
            nn.Linear(encoder_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Alpha prediction head
        self.alpha_head = nn.Sequential(
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(head_hidden_dim // 2, 1),
        )
    
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (batch, T, 2) normalized positions
            
        Returns:
            alpha: (batch,) in [0.1, 2.0]
        """
        # Encode
        features = self.encoder(trajectory)
        shared = self.shared_fc(features)
        
        # Predict α: sigmoid → [0.1, 2.0]
        alpha_raw = self.alpha_head(shared).squeeze(-1)
        alpha = torch.sigmoid(alpha_raw) * 1.9 + 0.1
        
        return alpha
    
    def get_config(self) -> dict:
        """Return model configuration for checkpointing."""
        return {
            "encoder_name": self.encoder_name,
            "encoder_output_dim": self.encoder.output_dim,
        }


class DiffusionPredictorInference(InferenceMethod):
    """
    Inference wrapper for trained DiffusionPredictor models.
    
    Conforms to the InferenceMethod interface for comparison with baselines.
    """
    
    def __init__(
        self,
        model: Optional[DiffusionPredictor] = None,
        checkpoint_path: Optional[str] = None,
        encoder_name: str = "lstm",
        device: str = "cpu",
    ):
        """
        Args:
            model: Pre-trained model (if None, will load from checkpoint)
            checkpoint_path: Path to saved checkpoint
            encoder_name: Encoder type (used for naming)
            device: Device for inference
        """
        # Create display name based on encoder
        name = f"{encoder_name.upper()} Predictor"
        super().__init__(name=name)
        
        self.device = device
        self.encoder_name = encoder_name
        
        if model is not None:
            self.model = model.to(device)
        elif checkpoint_path is not None:
            self.model = self._load_checkpoint(checkpoint_path)
        else:
            raise ValueError("Must provide either model or checkpoint_path")
        
        self.model.eval()
    
    def _load_checkpoint(self, path: str) -> DiffusionPredictor:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        config = checkpoint.get("config", {})
        encoder_name = config.get("encoder_name", self.encoder_name)
        
        model = DiffusionPredictor(
            encoder_name=encoder_name,
            encoder_kwargs=config.get("encoder_kwargs", {}),
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)
    
    def fit(self, trajectory: np.ndarray) -> None:
        """
        Run inference on a trajectory.
        
        Note: This doesn't train - uses pre-trained model.
        """
        self.reset()
        
        # Convert to tensor
        traj_tensor = torch.tensor(
            trajectory, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            alpha = self.model(traj_tensor)
        
        self._alpha = alpha.item()
        self._is_fitted = True


def create_predictor(
    encoder_name: str = "lstm",
    **encoder_kwargs,
) -> DiffusionPredictor:
    """Factory function to create a predictor with specified encoder."""
    return DiffusionPredictor(
        encoder_name=encoder_name,
        encoder_kwargs=encoder_kwargs,
    )
