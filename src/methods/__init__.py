"""
Methods for anomalous diffusion parameter estimation.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w):
- Task 1: α inference (anomalous diffusion exponent)
- Task 2: Model classification (CTRW, FBM, LW, ATTM, SBM)

Architecture:
    trajectory (batch, T, 2) → Encoder → features → Heads → (α, model_logits)

Components:
- Encoders: linear, mlp, lstm, cnn, cnn-lstm
- Losses: supervised, physics, combined, classification, multitask
- Predictor: combines encoder + heads (supports single/multi-task)

The "PINN" aspect is just using the physics loss - any encoder can be a PINN!
"""

# Base class
from .base import InferenceMethod

# Per-trajectory baselines (for comparison)
from .msd_fitting import MSDFitting
from .msd_pinn import MSDPINN
from .displacement_pinn import DisplacementPINN

# Modular components
from .encoders import (
    LinearEncoder,
    MLPEncoder,
    LSTMEncoder,
    CNNEncoder,
    CNNLSTMEncoder,
    create_encoder,
    ENCODERS,
)

from .losses import (
    SupervisedLoss,
    PhysicsLoss,
    CombinedLoss,
    ModelClassificationLoss,
    MultiTaskLoss,
    create_loss,
    LOSSES,
)

from .predictor import (
    DiffusionPredictor,
    DiffusionPredictorInference,
    create_predictor,
)

__all__ = [
    # Base
    "InferenceMethod",
    
    # Per-trajectory baselines
    "MSDFitting",
    "MSDPINN",
    "DisplacementPINN",
    
    # Encoders
    "LinearEncoder",
    "MLPEncoder", 
    "LSTMEncoder",
    "CNNEncoder",
    "CNNLSTMEncoder",
    "create_encoder",
    "ENCODERS",
    
    # Losses
    "SupervisedLoss",
    "PhysicsLoss",
    "CombinedLoss",
    "ModelClassificationLoss",
    "MultiTaskLoss",
    "create_loss",
    "LOSSES",
    
    # Main predictor
    "DiffusionPredictor",
    "DiffusionPredictorInference",
    "create_predictor",
]
