"""
Evaluation metrics and comparison utilities.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w),
we evaluate methods on:
- Task 1: α (anomalous diffusion exponent) prediction
- Task 2: Diffusion model classification (CTRW, FBM, LW, ATTM, SBM)

Extended metrics for direct comparison with AnDi paper results:
- MAE: Mean Absolute Error (primary metric for Task 1)
- RMSE: Root Mean Squared Error
- Bias: Mean signed error (should be ~0)
- R²: Coefficient of determination
- Percentile errors (P90, P95)
- F1-score: Micro-averaged F1 (primary metric for Task 2)
- Confusion Matrix: Per-class breakdown
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.trajectory import Trajectory, TrajectoryDataset, MODEL_NAMES, NUM_MODELS
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


# =============================================================================
# Task 2: Model Classification Metrics
# =============================================================================


@dataclass
class ClassificationMetrics:
    """
    Classification metrics for Task 2 (diffusion model classification).
    
    Following the AnDi Challenge convention, the primary metric is micro-averaged F1.
    """
    
    n_samples: int = 0
    
    # Overall metrics
    accuracy: float = 0.0
    f1_micro: float = 0.0      # Primary metric (per ANDI)
    f1_macro: float = 0.0      # Average across classes
    f1_weighted: float = 0.0   # Weighted by class support
    
    # Per-class metrics
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    
    # Confusion matrix (as nested dict for JSON serialization)
    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None
    
    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "f1_micro": self.f1_micro,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "per_class_accuracy": self.per_class_accuracy,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "per_class_support": self.per_class_support,
            "confusion_matrix": self.confusion_matrix,
        }
    
    def __str__(self) -> str:
        return (
            f"ClassificationMetrics(n={self.n_samples}, "
            f"accuracy={self.accuracy:.4f}, F1_micro={self.f1_micro:.4f})"
        )


def compute_confusion_matrix(
    model_true: np.ndarray,
    model_pred: np.ndarray,
    num_classes: int = NUM_MODELS,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        model_true: Ground truth class indices
        model_pred: Predicted class indices
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        cm[i, j] = number of samples with true label i predicted as j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(model_true, model_pred):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            cm[int(true), int(pred)] += 1
    return cm


def compute_classification_metrics(
    model_true: np.ndarray,
    model_pred: np.ndarray,
    class_names: List[str] = None,
) -> ClassificationMetrics:
    """
    Compute all classification metrics for Task 2.
    
    Args:
        model_true: Ground truth class indices (0-4)
        model_pred: Predicted class indices (0-4)
        class_names: List of class names (default: MODEL_NAMES)
        
    Returns:
        ClassificationMetrics with all computed values
    """
    if class_names is None:
        class_names = MODEL_NAMES
    
    num_classes = len(class_names)
    
    model_true = np.asarray(model_true)
    model_pred = np.asarray(model_pred)
    
    # Filter invalid labels
    valid_mask = (model_true >= 0) & (model_true < num_classes)
    valid_mask &= (model_pred >= 0) & (model_pred < num_classes)
    
    model_true = model_true[valid_mask]
    model_pred = model_pred[valid_mask]
    
    n = len(model_true)
    if n == 0:
        return ClassificationMetrics()
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(model_true, model_pred, num_classes)
    
    # Overall accuracy
    accuracy = np.trace(cm) / n
    
    # Per-class metrics
    per_class_accuracy = {}
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    per_class_support = {}
    
    # For micro averaging
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # For macro/weighted averaging
    f1_scores = []
    weights = []
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp  # Predicted as i but not i
        fn = cm[i, :].sum() - tp  # Actually i but not predicted as i
        tn = n - tp - fp - fn
        
        support = cm[i, :].sum()  # Total actual samples of class i
        per_class_support[name] = int(support)
        
        # Accuracy for this class vs all others
        per_class_accuracy[name] = float((tp + tn) / n) if n > 0 else 0.0
        
        # Precision
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        per_class_precision[name] = precision
        
        # Recall
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        per_class_recall[name] = recall
        
        # F1
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_f1[name] = f1
        
        if support > 0:
            f1_scores.append(f1)
            weights.append(support)
        
        # Accumulate for micro averaging
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Micro-averaged F1 (sum all TP, FP, FN then compute)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Macro-averaged F1 (average of per-class F1)
    f1_macro = float(np.mean(f1_scores)) if f1_scores else 0.0
    
    # Weighted F1 (weighted by support)
    f1_weighted = float(np.average(f1_scores, weights=weights)) if f1_scores and sum(weights) > 0 else 0.0
    
    # Convert confusion matrix to dict for JSON serialization
    cm_dict = {}
    for i, true_name in enumerate(class_names):
        cm_dict[true_name] = {}
        for j, pred_name in enumerate(class_names):
            cm_dict[true_name][pred_name] = int(cm[i, j])
    
    return ClassificationMetrics(
        n_samples=n,
        accuracy=float(accuracy),
        f1_micro=float(f1_micro),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        per_class_accuracy=per_class_accuracy,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        confusion_matrix=cm_dict,
    )


def print_confusion_matrix(
    cm: Dict[str, Dict[str, int]],
    class_names: List[str] = None,
) -> None:
    """Print a formatted confusion matrix."""
    if class_names is None:
        class_names = list(cm.keys())
    
    # Header
    print("\nConfusion Matrix:")
    print("=" * (15 + 10 * len(class_names)))
    header = "True \\ Pred".ljust(14)
    for name in class_names:
        header += f"{name:>10}"
    print(header)
    print("-" * (15 + 10 * len(class_names)))
    
    # Rows
    for true_name in class_names:
        row = f"{true_name:<14}"
        for pred_name in class_names:
            val = cm.get(true_name, {}).get(pred_name, 0)
            row += f"{val:>10}"
        print(row)
    
    print("=" * (15 + 10 * len(class_names)))


def print_classification_summary(metrics: ClassificationMetrics) -> None:
    """Print a formatted classification metrics summary."""
    print("\n" + "=" * 70)
    print("Classification Metrics (Task 2)")
    print("=" * 70)
    print(f"Total samples: {metrics.n_samples}")
    print(f"Overall accuracy: {metrics.accuracy:.4f}")
    print(f"F1-score (micro): {metrics.f1_micro:.4f}  ← Primary metric (per ANDI)")
    print(f"F1-score (macro): {metrics.f1_macro:.4f}")
    print(f"F1-score (weighted): {metrics.f1_weighted:.4f}")
    
    print("\nPer-class breakdown:")
    print("-" * 70)
    print(f"{'Model':<10} {'Support':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    
    for name in MODEL_NAMES:
        if name in metrics.per_class_support:
            support = metrics.per_class_support[name]
            precision = metrics.per_class_precision.get(name, 0)
            recall = metrics.per_class_recall.get(name, 0)
            f1 = metrics.per_class_f1.get(name, 0)
            print(f"{name:<10} {support:>10} {precision:>12.4f} {recall:>10.4f} {f1:>10.4f}")
    
    print("=" * 70)
    
    if metrics.confusion_matrix:
        print_confusion_matrix(metrics.confusion_matrix)


@dataclass 
class ModelResult:
    """Result of model classification on a single trajectory."""
    
    trajectory_id: int
    model_true: int
    model_pred: int
    method_name: str
    length: Optional[int] = None
    
    @property
    def is_correct(self) -> bool:
        """Whether the prediction is correct."""
        return self.model_pred == self.model_true
    
    @property
    def model_true_name(self) -> str:
        """Model name for ground truth."""
        return MODEL_NAMES[self.model_true] if 0 <= self.model_true < NUM_MODELS else "Unknown"
    
    @property
    def model_pred_name(self) -> str:
        """Model name for prediction."""
        return MODEL_NAMES[self.model_pred] if 0 <= self.model_pred < NUM_MODELS else "Unknown"
    
    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "model_true": self.model_true,
            "model_pred": self.model_pred,
            "model_true_name": self.model_true_name,
            "model_pred_name": self.model_pred_name,
            "is_correct": self.is_correct,
            "method_name": self.method_name,
            "length": self.length,
        }
