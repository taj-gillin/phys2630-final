"""Evaluation metrics and comparison utilities."""

from dataclasses import dataclass
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
    D0_true: Optional[float]
    D0_pred: float
    method_name: str
    
    @property
    def alpha_error(self) -> float:
        """Absolute error in alpha."""
        return abs(self.alpha_pred - self.alpha_true)
    
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
            "alpha_relative_error": self.alpha_relative_error,
            "D0_true": self.D0_true,
            "D0_pred": self.D0_pred,
            "method_name": self.method_name,
        }


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
    
    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "n_trajectories": self.n_trajectories,
            "mean_alpha_error": self.mean_alpha_error,
            "std_alpha_error": self.std_alpha_error,
            "mean_relative_error": self.mean_relative_error,
            "std_relative_error": self.std_relative_error,
            "median_alpha_error": self.median_alpha_error,
        }


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
            alpha_pred, D0_pred = method.fit_predict(traj.positions)
        except Exception as e:
            if verbose:
                print(f"  Error on trajectory {i}: {e}")
            alpha_pred = np.nan
            D0_pred = np.nan
        
        result = TrajectoryResult(
            trajectory_id=traj.trajectory_id or i,
            alpha_true=traj.alpha_true,
            alpha_pred=alpha_pred,
            D0_true=traj.D0_true,
            D0_pred=D0_pred,
            method_name=method.name,
        )
        results.append(result)
    
    return results


def summarize_results(results: list[TrajectoryResult]) -> MethodSummary:
    """
    Compute summary statistics from trajectory results.
    
    Args:
        results: List of TrajectoryResult from same method
        
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
    
    errors = [r.alpha_error for r in valid_results]
    rel_errors = [r.alpha_relative_error for r in valid_results]
    
    return MethodSummary(
        method_name=results[0].method_name,
        n_trajectories=len(results),
        mean_alpha_error=np.mean(errors),
        std_alpha_error=np.std(errors),
        mean_relative_error=np.mean(rel_errors),
        std_relative_error=np.std(rel_errors),
        median_alpha_error=np.median(errors),
    )


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


def print_comparison_table(summaries: dict[str, MethodSummary]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("Method Comparison Summary")
    print("=" * 70)
    print(f"{'Method':<25} {'Mean Error':>12} {'Std Error':>12} {'Rel Error':>12}")
    print("-" * 70)
    
    for name, summary in summaries.items():
        print(f"{name:<25} {summary.mean_alpha_error:>12.4f} {summary.std_alpha_error:>12.4f} {summary.mean_relative_error:>11.2%}")
    
    print("=" * 70)




