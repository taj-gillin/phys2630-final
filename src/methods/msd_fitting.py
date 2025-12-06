"""
Pure MSD fitting baseline for anomalous diffusion exponent inference.

Following the AnDi Challenge conventions, we fit:
    log(MSD) = α·log(τ) + const

to extract α from the slope of the log-log MSD plot.
"""

import numpy as np
from scipy.stats import linregress

from .base import InferenceMethod


class MSDFitting(InferenceMethod):
    """
    Traditional MSD-based estimation of anomalous diffusion exponent.
    
    Fits log(MSD) = α·log(τ) + const via linear regression.
    """
    
    def __init__(self, max_lag_fraction: float = 0.25):
        """
        Args:
            max_lag_fraction: Maximum lag time as fraction of trajectory length
        """
        super().__init__(name="MSD Fitting")
        self.max_lag_fraction = max_lag_fraction
        self._r_squared: float = 0.0
    
    def fit(self, trajectory: np.ndarray) -> None:
        """
        Fit alpha from trajectory MSD.
        
        Args:
            trajectory: Array of shape (T, 2) with (x, y) positions
        """
        self.reset()
        
        # Compute MSD
        lags, msd = self._compute_msd(trajectory)
        
        # Filter valid points (positive MSD values)
        mask = (lags > 0) & (msd > 0)
        if np.sum(mask) < 3:
            # Not enough points for regression
            self._alpha = np.nan
            self._r_squared = 0.0
            self._is_fitted = True
            return
        
        # Linear regression on log-log scale
        log_lags = np.log(lags[mask])
        log_msd = np.log(msd[mask])
        
        slope, intercept, r, _, _ = linregress(log_lags, log_msd)
        
        # MSD ∝ τ^α  =>  log(MSD) = α·log(τ) + const
        self._alpha = slope
        self._r_squared = r ** 2
        self._is_fitted = True
    
    def _compute_msd(self, trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean squared displacement for multiple lag times."""
        T = len(trajectory)
        max_lag = max(1, int(T * self.max_lag_fraction))
        lags = np.arange(1, max_lag + 1)
        msd = np.zeros(len(lags))
        
        for i, lag in enumerate(lags):
            disps = trajectory[lag:] - trajectory[:-lag]
            msd[i] = np.mean(np.sum(disps ** 2, axis=-1))
        
        return lags, msd
    
    @property
    def r_squared(self) -> float:
        """R-squared of the log-log fit."""
        return self._r_squared
