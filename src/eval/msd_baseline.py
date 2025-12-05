import numpy as np
from scipy.stats import linregress
from typing import List, Tuple


def compute_msd(traj: np.ndarray, max_lag_frac: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    T = len(traj)
    max_lag = max(1, int(T * max_lag_frac))
    lags = np.arange(1, max_lag + 1)
    msd = np.zeros_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        disps = traj[lag:] - traj[:-lag]
        msd[i] = np.mean(np.sum(disps ** 2, axis=-1))
    return lags, msd


def fit_alpha_msd(lags: np.ndarray, msd: np.ndarray) -> Tuple[float, float, float]:
    mask = (lags > 0) & (msd > 0)
    log_lags = np.log(lags[mask])
    log_msd = np.log(msd[mask])
    slope, intercept, r, _, _ = linregress(log_lags, log_msd)
    alpha = slope
    K = np.exp(intercept) / 4  # for 2D MSD = 4 K t^alpha
    return alpha, K, r ** 2


def fit_trajectories_msd(trajs: List[np.ndarray]) -> Tuple[float, float]:
    alphas = []
    for traj in trajs:
        lags, msd = compute_msd(traj)
        alpha, _, _ = fit_alpha_msd(lags, msd)
        if not np.isnan(alpha):
            alphas.append(alpha)
    if len(alphas) == 0:
        return np.nan, np.nan
    return float(np.mean(alphas)), float(np.std(alphas))

