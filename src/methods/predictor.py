"""
Unified Diffusion Predictor: combines any encoder with prediction heads.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w),
this model supports:
- Task 1: Predict α (anomalous diffusion exponent)
- Task 2: Classify the underlying diffusion model (5 classes)

This is the main model class that:
1. Takes a trajectory as input
2. Uses an encoder to extract features
3. Predicts α (Task 1) and/or model class (Task 2)

The encoder and loss are modular - mix and match as needed.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List

from .base import InferenceMethod
from .encoders import create_encoder, ENCODERS
from .losses import create_loss, LOSSES

# Number of diffusion models (CTRW, FBM, LW, ATTM, SBM)
NUM_MODELS = 5


class DiffusionPredictor(nn.Module):
    """
    Universal model for predicting α and/or diffusion model from trajectories.
    
    Architecture:
        trajectory → encoder → features → shared_fc → {alpha_head, model_head}
    
    The encoder is pluggable (Linear, MLP, LSTM, CNN, CNN-LSTM).
    
    Tasks can be enabled/disabled:
        - tasks=['alpha']:        Task 1 only (original behavior)
        - tasks=['model']:        Task 2 only
        - tasks=['alpha','model']: Both tasks (multi-task learning)
    """
    
    def __init__(
        self,
        encoder_name: str = "lstm",
        encoder_kwargs: dict = None,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        num_models: int = NUM_MODELS,
        tasks: List[str] = None,
    ):
        """
        Args:
            encoder_name: One of 'linear', 'mlp', 'lstm', 'cnn', 'cnn-lstm'
            encoder_kwargs: Arguments passed to encoder constructor
            head_hidden_dim: Hidden dimension for prediction heads
            dropout: Dropout rate
            num_models: Number of diffusion model classes (default: 5)
            tasks: List of tasks to perform. Options: ['alpha', 'model']. 
                   Default: ['alpha'] for backward compatibility.
        """
        super().__init__()
        
        self.encoder_name = encoder_name
        self.num_models = num_models
        self.tasks = tasks if tasks is not None else ['alpha']
        encoder_kwargs = encoder_kwargs or {}
        
        # Validate tasks
        valid_tasks = {'alpha', 'model'}
        for task in self.tasks:
            if task not in valid_tasks:
                raise ValueError(f"Unknown task: {task}. Valid tasks: {valid_tasks}")
        
        # Create encoder
        self.encoder = create_encoder(encoder_name, **encoder_kwargs)
        encoder_dim = self.encoder.output_dim
        
        # Shared feature refinement
        self.shared_fc = nn.Sequential(
            nn.Linear(encoder_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Alpha prediction head (Task 1)
        if 'alpha' in self.tasks:
            self.alpha_head = nn.Sequential(
                nn.Linear(head_hidden_dim, head_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(head_hidden_dim // 2, 1),
            )
        else:
            self.alpha_head = None
        
        # Model classification head (Task 2)
        if 'model' in self.tasks:
            self.model_head = nn.Sequential(
                nn.Linear(head_hidden_dim, head_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(head_hidden_dim // 2, num_models),
            )
        else:
            self.model_head = None
    
    def forward(
        self, 
        trajectory: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            trajectory: (batch, T, 2) normalized positions
            
        Returns:
            If single task:
                alpha: (batch,) in [0.1, 2.0] for Task 1
                OR
                model_logits: (batch, num_models) for Task 2
            If multi-task:
                Tuple of (alpha, model_logits)
        """
        # Encode
        features = self.encoder(trajectory)
        shared = self.shared_fc(features)
        
        outputs = {}
        
        # Predict α: sigmoid → [0.1, 2.0]
        if self.alpha_head is not None:
            alpha_raw = self.alpha_head(shared).squeeze(-1)
            alpha = torch.sigmoid(alpha_raw) * 1.9 + 0.1
            outputs['alpha'] = alpha
        
        # Predict model class (raw logits)
        if self.model_head is not None:
            model_logits = self.model_head(shared)
            outputs['model'] = model_logits
        
        # Return based on tasks
        if len(self.tasks) == 1:
            return outputs[self.tasks[0]]
        else:
            # Multi-task: return (alpha, model_logits)
            return outputs.get('alpha'), outputs.get('model')
    
    def predict_model(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Predict the diffusion model class.
        
        Args:
            trajectory: (batch, T, 2) normalized positions
            
        Returns:
            model_pred: (batch,) predicted model indices
        """
        if self.model_head is None:
            raise ValueError("Model classification head not enabled. Set tasks=['model'] or tasks=['alpha','model']")
        
        self.eval()
        with torch.no_grad():
            features = self.encoder(trajectory)
            shared = self.shared_fc(features)
            logits = self.model_head(shared)
            return torch.argmax(logits, dim=-1)
    
    def get_config(self) -> dict:
        """Return model configuration for checkpointing."""
        return {
            "encoder_name": self.encoder_name,
            "encoder_output_dim": self.encoder.output_dim,
            "num_models": self.num_models,
            "tasks": self.tasks,
        }


class DiffusionPredictorInference(InferenceMethod):
    """
    Inference wrapper for trained DiffusionPredictor models.
    
    Conforms to the InferenceMethod interface for comparison with baselines.
    Supports both Task 1 (α inference) and Task 2 (model classification).
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
        self._model_pred = None  # Store model prediction
        
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
            num_models=config.get("num_models", NUM_MODELS),
            tasks=config.get("tasks", ["alpha"]),
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)
    
    @property
    def model_pred(self) -> Optional[int]:
        """Get predicted model index (Task 2)."""
        return self._model_pred
    
    def reset(self):
        """Reset inference state."""
        super().reset()
        self._model_pred = None
    
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
            output = self.model(traj_tensor)
            
            # Handle single-task vs multi-task output
            if isinstance(output, tuple):
                alpha, model_logits = output
                if alpha is not None:
                    self._alpha = alpha.item()
                if model_logits is not None:
                    self._model_pred = torch.argmax(model_logits, dim=-1).item()
            else:
                # Single task
                if 'alpha' in self.model.tasks:
                    self._alpha = output.item()
                elif 'model' in self.model.tasks:
                    self._model_pred = torch.argmax(output, dim=-1).item()
        
        self._is_fitted = True


def create_predictor(
    encoder_name: str = "lstm",
    tasks: List[str] = None,
    **encoder_kwargs,
) -> DiffusionPredictor:
    """Factory function to create a predictor with specified encoder."""
    return DiffusionPredictor(
        encoder_name=encoder_name,
        encoder_kwargs=encoder_kwargs,
        tasks=tasks,
    )
