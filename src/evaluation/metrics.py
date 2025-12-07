"""
Evaluation metrics and comparison utilities.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w),
we evaluate methods on α (anomalous diffusion exponent) prediction.

Extended metrics for direct comparison with AnDi paper results:
- MAE: Mean Absolute Error (primary metric)
- RMSE: Root Mean Squared Error
- Bias: Mean signed error (should be ~0)
- R²: Coefficient of determination
- Percentile errors (P90, P95)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.trajectory import Trajectory, TrajectoryDataset
from methods.base import InferenceMethod


@dataclass
class TrajectoryResult:
    """Result of inference on a single trajectory."""
    
    trajectory_id: int
    alpha_true: float
    alpha_pred: float
    method_name: str
    length: Optional[int] = None
    
    @property
    def alpha_error(self) -> float:
        """Absolute error in alpha."""
        return abs(self.alpha_pred - self.alpha_true)
    
    @property
    def alpha_bias(self) -> float:
        """Signed error (bias) in alpha."""
        return self.alpha_pred - self.alpha_true
    
    @property
    def alpha_relative_error(self) -> float:
        """Relative error in alpha."""
        if self.alpha_true == 0:
            return float('inf')
        return abs(self.alpha_pred - self.alpha_true) / abs(self.alpha_true)
    
    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "alpha_true": self.alpha_true,
            "alpha_pred": self.alpha_pred,
            "alpha_error": self.alpha_error,
            "alpha_bias": self.alpha_bias,
            "alpha_relative_error": self.alpha_relative_error,
            "method_name": self.method_name,
            "length": self.length,
        }


@dataclass
class ExtendedMetrics:
    """
    Extended metrics for AnDi-style comparison.
    
    These metrics match what the AnDi Challenge paper reports,
    enabling direct comparison with published results.
    """
    
    # Sample info
    n_samples: int = 0
    
    # Primary error metrics (AnDi uses MAE)
    mae: float = 0.0           # Mean Absolute Error
    rmse: float = 0.0          # Root Mean Squared Error
    mse: float = 0.0           # Mean Squared Error
    median_ae: float = 0.0     # Median Absolute Error (robust)
    
    # Percentile errors (for worst-case analysis)
    p90_error: float = 0.0     # 90th percentile error
    p95_error: float = 0.0     # 95th percentile error
    max_error: float = 0.0     # Maximum error
    
    # Bias analysis (AnDi checks for systematic bias)
    bias: float = 0.0          # Mean(pred - true), should be ~0
    bias_std: float = 0.0      # Std of bias
    
    # Correlation metrics
    r2: float = 0.0            # R² score
    pearson_r: float = 0.0     # Pearson correlation coefficient
    
    # Relative errors
    mean_rel_error: float = 0.0
    median_rel_error: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "mae": self.mae,
            "rmse": self.rmse,
            "mse": self.mse,
            "median_ae": self.median_ae,
            "p90_error": self.p90_error,
            "p95_error": self.p95_error,
            "max_error": self.max_error,
            "bias": self.bias,
            "bias_std": self.bias_std,
            "r2": self.r2,
            "pearson_r": self.pearson_r,
            "mean_rel_error": self.mean_rel_error,
            "median_rel_error": self.median_rel_error,
        }
    
    def __str__(self) -> str:
        return (
            f"ExtendedMetrics(n={self.n_samples}, MAE={self.mae:.4f}, "
            f"RMSE={self.rmse:.4f}, R²={self.r2:.4f}, bias={self.bias:.4f})"
        )


def compute_extended_metrics(
    alpha_true: np.ndarray,
    alpha_pred: np.ndarray,
) -> ExtendedMetrics:
    """
    Compute all extended metrics from predictions.
    
    Args:
        alpha_true: Ground truth α values
        alpha_pred: Predicted α values
        
    Returns:
        ExtendedMetrics with all computed values
    """
    # Convert to numpy if needed
    alpha_true = np.asarray(alpha_true)
    alpha_pred = np.asarray(alpha_pred)
    
    # Filter NaNs
    mask = ~(np.isnan(alpha_true) | np.isnan(alpha_pred))
    alpha_true = alpha_true[mask]
    alpha_pred = alpha_pred[mask]
    
    n = len(alpha_true)
    if n == 0:
        return ExtendedMetrics()
    
    # Errors
    errors = np.abs(alpha_pred - alpha_true)
    signed_errors = alpha_pred - alpha_true
    
    # Relative errors (avoid div by zero)
    rel_errors = errors / np.maximum(np.abs(alpha_true), 1e-8)
    
    # R² score
    ss_res = np.sum((alpha_true - alpha_pred) ** 2)
    ss_tot = np.sum((alpha_true - np.mean(alpha_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Pearson correlation
    if np.std(alpha_true) > 0 and np.std(alpha_pred) > 0:
        pearson_r = np.corrcoef(alpha_true, alpha_pred)[0, 1]
    else:
        pearson_r = 0.0
    
    return ExtendedMetrics(
        n_samples=n,
        mae=float(np.mean(errors)),
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mse=float(np.mean(errors ** 2)),
        median_ae=float(np.median(errors)),
        p90_error=float(np.percentile(errors, 90)),
        p95_error=float(np.percentile(errors, 95)),
        max_error=float(np.max(errors)),
        bias=float(np.mean(signed_errors)),
        bias_std=float(np.std(signed_errors)),
        r2=float(r2),
        pearson_r=float(pearson_r),
        mean_rel_error=float(np.mean(rel_errors)),
        median_rel_error=float(np.median(rel_errors)),
    )


@dataclass
class MethodSummary:
    """Summary statistics for a method across multiple trajectories."""
    
    method_name: str
    n_trajectories: int
    mean_alpha_error: float
    std_alpha_error: float
    mean_relative_error: float
    std_relative_error: float
    median_alpha_error: float
    
    # Extended metrics (optional, for AnDi comparison)
    rmse: Optional[float] = None
    bias: Optional[float] = None
    bias_std: Optional[float] = None
    r2: Optional[float] = None
    p90_error: Optional[float] = None
    p95_error: Optional[float] = None
    
    def to_dict(self) -> dict:
        d = {
            "method_name": self.method_name,
            "n_trajectories": self.n_trajectories,
            "mean_alpha_error": self.mean_alpha_error,
            "std_alpha_error": self.std_alpha_error,
            "mean_relative_error": self.mean_relative_error,
            "std_relative_error": self.std_relative_error,
            "median_alpha_error": self.median_alpha_error,
        }
        # Add extended metrics if available
        if self.rmse is not None:
            d["rmse"] = self.rmse
        if self.bias is not None:
            d["bias"] = self.bias
            d["bias_std"] = self.bias_std
        if self.r2 is not None:
            d["r2"] = self.r2
        if self.p90_error is not None:
            d["p90_error"] = self.p90_error
            d["p95_error"] = self.p95_error
        return d


def evaluate_method(
    method: InferenceMethod,
    dataset: TrajectoryDataset,
    verbose: bool = False,
) -> list[TrajectoryResult]:
    """
    Evaluate an inference method on a dataset.
    
    Args:
        method: Inference method to evaluate
        dataset: Dataset of trajectories with ground truth
        verbose: Print progress
        
    Returns:
        List of TrajectoryResult for each trajectory
    """
    results = []
    
    for i, traj in enumerate(dataset):
        if verbose and i % 10 == 0:
            print(f"  Processing trajectory {i}/{len(dataset)}")
        
        method.reset()
        try:
            alpha_pred = method.fit_predict(traj.positions)
        except Exception as e:
            if verbose:
                print(f"  Error on trajectory {i}: {e}")
            alpha_pred = np.nan
        
        result = TrajectoryResult(
            trajectory_id=traj.trajectory_id or i,
            alpha_true=traj.alpha_true,
            alpha_pred=alpha_pred,
            method_name=method.name,
            length=len(traj.positions),
        )
        results.append(result)
    
    return results


def summarize_results(results: list[TrajectoryResult], extended: bool = True) -> MethodSummary:
    """
    Compute summary statistics from trajectory results.
    
    Args:
        results: List of TrajectoryResult from same method
        extended: Whether to compute extended AnDi-style metrics
        
    Returns:
        MethodSummary with aggregate statistics
    """
    if not results:
        raise ValueError("No results to summarize")
    
    # Filter out NaN results
    valid_results = [r for r in results if not np.isnan(r.alpha_pred)]
    
    if not valid_results:
        return MethodSummary(
            method_name=results[0].method_name,
            n_trajectories=len(results),
            mean_alpha_error=np.nan,
            std_alpha_error=np.nan,
            mean_relative_error=np.nan,
            std_relative_error=np.nan,
            median_alpha_error=np.nan,
        )
    
    errors = np.array([r.alpha_error for r in valid_results])
    rel_errors = np.array([r.alpha_relative_error for r in valid_results])
    
    summary = MethodSummary(
        method_name=results[0].method_name,
        n_trajectories=len(results),
        mean_alpha_error=float(np.mean(errors)),
        std_alpha_error=float(np.std(errors)),
        mean_relative_error=float(np.mean(rel_errors)),
        std_relative_error=float(np.std(rel_errors)),
        median_alpha_error=float(np.median(errors)),
    )
    
    # Add extended metrics for AnDi comparison
    if extended:
        alpha_true = np.array([r.alpha_true for r in valid_results])
        alpha_pred = np.array([r.alpha_pred for r in valid_results])
        
        ext = compute_extended_metrics(alpha_true, alpha_pred)
        summary.rmse = ext.rmse
        summary.bias = ext.bias
        summary.bias_std = ext.bias_std
        summary.r2 = ext.r2
        summary.p90_error = ext.p90_error
        summary.p95_error = ext.p95_error
    
    return summary


def compare_methods(
    methods: list[InferenceMethod],
    dataset: TrajectoryDataset,
    verbose: bool = True,
) -> dict:
    """
    Compare multiple inference methods on the same dataset.
    
    Args:
        methods: List of inference methods to compare
        dataset: Dataset of trajectories
        verbose: Print progress
        
    Returns:
        Dictionary with results and summaries for each method
    """
    all_results = {}
    summaries = {}
    
    for method in methods:
        if verbose:
            print(f"\nEvaluating {method.name}...")
        
        results = evaluate_method(method, dataset, verbose=verbose)
        summary = summarize_results(results)
        
        all_results[method.name] = results
        summaries[method.name] = summary
        
        if verbose:
            print(f"  Mean α error: {summary.mean_alpha_error:.4f} ± {summary.std_alpha_error:.4f}")
            print(f"  Mean relative error: {summary.mean_relative_error:.2%}")
    
    return {
        "results": all_results,
        "summaries": summaries,
    }


def results_by_alpha(
    results: list[TrajectoryResult],
    alpha_values: list[float],
    tol: float = 0.01,
) -> dict[float, list[TrajectoryResult]]:
    """Group results by true alpha value."""
    grouped = {}
    for alpha in alpha_values:
        grouped[alpha] = [
            r for r in results
            if abs(r.alpha_true - alpha) < tol
        ]
    return grouped


def save_results(
    comparison: dict,
    output_dir: Path,
    filename: str = "comparison_results.json",
) -> None:
    """Save comparison results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    output = {
        "summaries": {
            name: summary.to_dict()
            for name, summary in comparison["summaries"].items()
        },
        "results": {
            name: [r.to_dict() for r in results]
            for name, results in comparison["results"].items()
        },
    }
    
    with open(output_dir / filename, "w") as f:
        json.dump(output, f, indent=2)


def print_comparison_table(summaries: dict[str, MethodSummary], extended: bool = True) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("Method Comparison Summary (AnDi-Style Metrics)")
    print("=" * 90)
    
    if extended and any(s.rmse is not None for s in summaries.values()):
        print(f"{'Method':<20} {'MAE':>10} {'RMSE':>10} {'Bias':>10} {'R²':>10} {'P90':>10}")
        print("-" * 90)
        
        # Sort by MAE
        sorted_items = sorted(summaries.items(), key=lambda x: x[1].mean_alpha_error)
        for name, s in sorted_items:
            rmse = f"{s.rmse:.4f}" if s.rmse is not None else "N/A"
            bias = f"{s.bias:.4f}" if s.bias is not None else "N/A"
            r2 = f"{s.r2:.4f}" if s.r2 is not None else "N/A"
            p90 = f"{s.p90_error:.4f}" if s.p90_error is not None else "N/A"
            print(f"{name:<20} {s.mean_alpha_error:>10.4f} {rmse:>10} {bias:>10} {r2:>10} {p90:>10}")
    else:
        print(f"{'Method':<25} {'MAE':>12} {'Std':>12} {'Rel Error':>12}")
        print("-" * 70)
        
        for name, summary in summaries.items():
            print(f"{name:<25} {summary.mean_alpha_error:>12.4f} {summary.std_alpha_error:>12.4f} {summary.mean_relative_error:>11.2%}")
    
    print("=" * 90)


def breakdown_by_alpha(
    results: list[TrajectoryResult],
    bins: Optional[list[tuple[float, float, str]]] = None,
) -> dict[str, ExtendedMetrics]:
    """
    Compute extended metrics broken down by α value ranges.
    
    Follows AnDi paper analysis of performance across different α regimes.
    
    Args:
        results: List of TrajectoryResult
        bins: List of (low, high, name) tuples for binning
        
    Returns:
        Dictionary mapping bin names to ExtendedMetrics
    """
    if bins is None:
        bins = [
            (0.0, 0.5, "subdiffusion_strong"),   # Strong subdiffusion
            (0.5, 0.8, "subdiffusion_weak"),     # Weak subdiffusion
            (0.8, 1.2, "normal_like"),           # Near-normal diffusion
            (1.2, 1.5, "superdiffusion_weak"),   # Weak superdiffusion
            (1.5, 2.0, "superdiffusion_strong"), # Strong superdiffusion
        ]
    
    breakdown = {}
    for low, high, name in bins:
        filtered = [r for r in results if low <= r.alpha_true < high]
        if filtered:
            alpha_true = np.array([r.alpha_true for r in filtered])
            alpha_pred = np.array([r.alpha_pred for r in filtered])
            breakdown[name] = compute_extended_metrics(alpha_true, alpha_pred)
    
    return breakdown


def breakdown_by_length(
    results: list[TrajectoryResult],
    bins: Optional[list[tuple[int, int, str]]] = None,
) -> dict[str, ExtendedMetrics]:
    """
    Compute extended metrics broken down by trajectory length.
    
    Follows AnDi paper analysis of performance vs trajectory length.
    
    Args:
        results: List of TrajectoryResult (must have length attribute)
        bins: List of (low, high, name) tuples for binning
        
    Returns:
        Dictionary mapping bin names to ExtendedMetrics
    """
    if bins is None:
        bins = [
            (0, 100, "short"),
            (100, 300, "medium"),
            (300, 600, "long"),
            (600, 10000, "very_long"),
        ]
    
    # Filter results that have length info
    results_with_length = [r for r in results if r.length is not None]
    if not results_with_length:
        return {}
    
    breakdown = {}
    for low, high, name in bins:
        filtered = [r for r in results_with_length if low <= r.length < high]
        if filtered:
            alpha_true = np.array([r.alpha_true for r in filtered])
            alpha_pred = np.array([r.alpha_pred for r in filtered])
            breakdown[name] = compute_extended_metrics(alpha_true, alpha_pred)
    
    return breakdown
