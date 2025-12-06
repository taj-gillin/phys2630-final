"""
Base class for inference methods.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w),
we focus on predicting α (anomalous diffusion exponent) only.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class InferenceMethod(ABC):
    """Abstract base class for anomalous diffusion exponent inference."""
    
    def __init__(self, name: str):
        self.name = name
        self._alpha: Optional[float] = None
        self._is_fitted: bool = False
    
    @abstractmethod
    def fit(self, trajectory: np.ndarray) -> None:
        """
        Fit the model to a single trajectory.
        
        Args:
            trajectory: Array of shape (T, 2) with (x, y) positions
        """
        pass
    
    def predict_alpha(self) -> float:
        """Return the inferred anomalous exponent."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if self._alpha is None:
            raise RuntimeError("Alpha was not estimated during fitting")
        return self._alpha
    
    def fit_predict(self, trajectory: np.ndarray) -> float:
        """
        Fit and return α in one call.
        
        Args:
            trajectory: Array of shape (T, 2) with (x, y) positions
            
        Returns:
            α (anomalous diffusion exponent)
        """
        self.fit(trajectory)
        return self.predict_alpha()
    
    def reset(self) -> None:
        """Reset the model to unfitted state."""
        self._alpha = None
        self._is_fitted = False
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
