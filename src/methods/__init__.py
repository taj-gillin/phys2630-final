"""
Methods for anomalous diffusion parameter estimation.

Architecture:
    trajectory (batch, T, 2) → Encoder → features → Heads → (α, D₀)

Components:
- Encoders: linear, mlp, lstm, cnn, hybrid
- Losses: supervised, physics, combined
- Predictor: combines encoder + heads

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
    HybridEncoder,
    create_encoder,
    ENCODERS,
)

from .losses import (
    SupervisedLoss,
    PhysicsLoss,
    CombinedLoss,
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
    "HybridEncoder",
    "create_encoder",
    "ENCODERS",
    
    # Losses
    "SupervisedLoss",
    "PhysicsLoss",
    "CombinedLoss",
    "create_loss",
    "LOSSES",
    
    # Main predictor
    "DiffusionPredictor",
    "DiffusionPredictorInference",
    "create_predictor",
]
