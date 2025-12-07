#!/usr/bin/env python3
"""
Analyze and visualize model results for comparison with AnDi Challenge.

This script provides comprehensive analysis including:
- Task 1: Extended metrics (MAE, RMSE, bias, R², percentiles) for α prediction
- Task 2: F1-score, confusion matrix for model classification
- Breakdown by α value ranges
- Breakdown by trajectory length
- Visualization of predictions vs ground truth

Usage:
    python scripts/analyze_results.py --checkpoint outputs/cnn/best.pt
    python scripts/analyze_results.py --checkpoint outputs/cnn/best.pt --output outputs/cnn/analysis
    python scripts/analyze_results.py --compare outputs/  # Compare all models
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.hf_loader import load_from_hf
from data.trajectory import MODEL_NAMES, NUM_MODELS


@dataclass
class ExtendedMetrics:
    """Extended metrics for AnDi-style comparison."""
    
    # Basic
    n_samples: int = 0
    
    # Error metrics
    mae: float = 0.0           # Mean Absolute Error (primary AnDi metric)
    rmse: float = 0.0          # Root Mean Squared Error
    mse: float = 0.0           # Mean Squared Error
    median_ae: float = 0.0     # Median Absolute Error (robust)
    
    # Percentile errors
    p90_error: float = 0.0     # 90th percentile error
    p95_error: float = 0.0     # 95th percentile error
    max_error: float = 0.0     # Maximum error
    
    # Bias analysis
    bias: float = 0.0          # Mean(pred - true), should be ~0
    bias_std: float = 0.0      # Std of bias
    
    # Correlation
    r2: float = 0.0            # R² score
    pearson_r: float = 0.0     # Pearson correlation
    
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


def compute_extended_metrics(
    alpha_true: np.ndarray,
    alpha_pred: np.ndarray,
) -> ExtendedMetrics:
    """Compute all extended metrics from predictions."""
    
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
        mae=np.mean(errors),
        rmse=np.sqrt(np.mean(errors ** 2)),
        mse=np.mean(errors ** 2),
        median_ae=np.median(errors),
        p90_error=np.percentile(errors, 90),
        p95_error=np.percentile(errors, 95),
        max_error=np.max(errors),
        bias=np.mean(signed_errors),
        bias_std=np.std(signed_errors),
        r2=r2,
        pearson_r=pearson_r,
        mean_rel_error=np.mean(rel_errors),
        median_rel_error=np.median(rel_errors),
    )


def breakdown_by_alpha(
    alpha_true: np.ndarray,
    alpha_pred: np.ndarray,
    bins: list[tuple[float, float, str]] = None,
) -> dict[str, ExtendedMetrics]:
    """Compute metrics broken down by α value ranges."""
    
    if bins is None:
        bins = [
            (0.0, 0.5, "subdiffusion_strong"),   # Strong subdiffusion
            (0.5, 0.8, "subdiffusion_weak"),     # Weak subdiffusion
            (0.8, 1.2, "normal_like"),           # Near-normal diffusion
            (1.2, 1.5, "superdiffusion_weak"),   # Weak superdiffusion
            (1.5, 2.0, "superdiffusion_strong"), # Strong superdiffusion
        ]
    
    results = {}
    for low, high, name in bins:
        mask = (alpha_true >= low) & (alpha_true < high)
        if np.sum(mask) > 0:
            results[name] = compute_extended_metrics(
                alpha_true[mask], alpha_pred[mask]
            )
            results[name].alpha_range = (low, high)
    
    return results


def breakdown_by_length(
    alpha_true: np.ndarray,
    alpha_pred: np.ndarray,
    lengths: np.ndarray,
    bins: list[tuple[int, int, str]] = None,
) -> dict[str, ExtendedMetrics]:
    """Compute metrics broken down by trajectory length."""
    
    if bins is None:
        bins = [
            (0, 100, "short"),
            (100, 300, "medium"),
            (300, 600, "long"),
            (600, 10000, "very_long"),
        ]
    
    results = {}
    for low, high, name in bins:
        mask = (lengths >= low) & (lengths < high)
        if np.sum(mask) > 0:
            results[name] = compute_extended_metrics(
                alpha_true[mask], alpha_pred[mask]
            )
            results[name].length_range = (low, high)
    
    return results


def create_visualizations(
    alpha_true: np.ndarray,
    alpha_pred: np.ndarray,
    lengths: np.ndarray,
    output_dir: Path,
    model_name: str = "Model",
):
    """Create visualization plots and save them."""
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available, skipping visualizations")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Scatter plot: Predicted vs True α
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 2D histogram for density
    h = ax.hist2d(alpha_true, alpha_pred, bins=50, cmap='Blues', 
                   norm=mcolors.LogNorm(), alpha=0.8)
    plt.colorbar(h[3], ax=ax, label='Count')
    
    # Perfect prediction line
    ax.plot([0, 2], [0, 2], 'r--', linewidth=2, label='Perfect prediction')
    
    # Compute and show metrics
    metrics = compute_extended_metrics(alpha_true, alpha_pred)
    textstr = f'MAE = {metrics.mae:.4f}\nRMSE = {metrics.rmse:.4f}\nR² = {metrics.r2:.4f}\nBias = {metrics.bias:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('True α', fontsize=12)
    ax.set_ylabel('Predicted α', fontsize=12)
    ax.set_title(f'{model_name}: Predicted vs True α', fontsize=14)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_pred_vs_true.png', dpi=150)
    plt.close()
    
    # 2. Bias histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bias = alpha_pred - alpha_true
    ax.hist(bias, bins=100, density=True, alpha=0.7, color='steelblue', 
            edgecolor='white', linewidth=0.5)
    
    # Overlay normal distribution
    from scipy import stats
    x = np.linspace(bias.min(), bias.max(), 100)
    ax.plot(x, stats.norm.pdf(x, metrics.bias, metrics.bias_std), 
            'r-', linewidth=2, label=f'Normal fit (μ={metrics.bias:.4f}, σ={metrics.bias_std:.4f})')
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero bias')
    ax.axvline(metrics.bias, color='red', linestyle='-', linewidth=1.5, label=f'Mean bias = {metrics.bias:.4f}')
    
    ax.set_xlabel('Prediction Bias (α_pred - α_true)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{model_name}: Bias Distribution', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_histogram.png', dpi=150)
    plt.close()
    
    # 3. Error vs α value
    fig, ax = plt.subplots(figsize=(10, 6))
    
    errors = np.abs(alpha_pred - alpha_true)
    
    # Bin errors by α value
    alpha_bins = np.linspace(0.1, 2.0, 20)
    bin_centers = (alpha_bins[:-1] + alpha_bins[1:]) / 2
    bin_means = []
    bin_stds = []
    
    for i in range(len(alpha_bins) - 1):
        mask = (alpha_true >= alpha_bins[i]) & (alpha_true < alpha_bins[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(errors[mask]))
            bin_stds.append(np.std(errors[mask]))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', 
                capsize=3, capthick=1, color='steelblue', linewidth=2, markersize=6)
    
    ax.axhline(metrics.mae, color='red', linestyle='--', linewidth=1.5, 
               label=f'Overall MAE = {metrics.mae:.4f}')
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7, 
               label='Normal diffusion (α=1)')
    
    ax.set_xlabel('True α', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'{model_name}: Error vs α Value', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_alpha.png', dpi=150)
    plt.close()
    
    # 4. Error vs trajectory length
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bin errors by length
    length_bins = [50, 100, 200, 300, 500, 750, 1000]
    bin_centers_len = [(length_bins[i] + length_bins[i+1]) / 2 for i in range(len(length_bins)-1)]
    bin_means_len = []
    bin_stds_len = []
    
    for i in range(len(length_bins) - 1):
        mask = (lengths >= length_bins[i]) & (lengths < length_bins[i+1])
        if np.sum(mask) > 0:
            bin_means_len.append(np.mean(errors[mask]))
            bin_stds_len.append(np.std(errors[mask]))
        else:
            bin_means_len.append(np.nan)
            bin_stds_len.append(np.nan)
    
    ax.errorbar(bin_centers_len, bin_means_len, yerr=bin_stds_len, fmt='s-',
                capsize=3, capthick=1, color='forestgreen', linewidth=2, markersize=8)
    
    ax.axhline(metrics.mae, color='red', linestyle='--', linewidth=1.5,
               label=f'Overall MAE = {metrics.mae:.4f}')
    
    ax.set_xlabel('Trajectory Length', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title(f'{model_name}: Error vs Trajectory Length', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_length.png', dpi=150)
    plt.close()
    
    # 5. Residual plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    residuals = alpha_pred - alpha_true
    ax.scatter(alpha_true, residuals, alpha=0.3, s=10, c='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    
    # Add trend line
    z = np.polyfit(alpha_true, residuals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(alpha_true.min(), alpha_true.max(), 100)
    ax.plot(x_line, p(x_line), 'orange', linewidth=2, 
            label=f'Trend: {z[0]:.4f}α + {z[1]:.4f}')
    
    ax.set_xlabel('True α', fontsize=12)
    ax.set_ylabel('Residual (α_pred - α_true)', fontsize=12)
    ax.set_title(f'{model_name}: Residual Plot', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=150)
    plt.close()
    
    print(f"  Saved visualizations to {output_dir}")


@dataclass
class ClassificationMetrics:
    """Classification metrics for Task 2."""
    n_samples: int = 0
    accuracy: float = 0.0
    f1_micro: float = 0.0
    f1_macro: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None
    
    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "f1_micro": self.f1_micro,
            "f1_macro": self.f1_macro,
            "per_class_f1": self.per_class_f1,
            "per_class_support": self.per_class_support,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_classification_metrics(
    model_true: np.ndarray,
    model_pred: np.ndarray,
    class_names: List[str] = None,
) -> ClassificationMetrics:
    """Compute classification metrics for Task 2."""
    if class_names is None:
        class_names = MODEL_NAMES
    
    num_classes = len(class_names)
    
    # Filter invalid
    valid_mask = (model_true >= 0) & (model_true < num_classes)
    valid_mask &= (model_pred >= 0) & (model_pred < num_classes)
    model_true = model_true[valid_mask]
    model_pred = model_pred[valid_mask]
    
    n = len(model_true)
    if n == 0:
        return ClassificationMetrics()
    
    # Confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(model_true, model_pred):
        cm[int(true), int(pred)] += 1
    
    accuracy = np.trace(cm) / n
    
    # Per-class metrics
    per_class_f1 = {}
    per_class_support = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    f1_scores = []
    weights = []
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        per_class_support[name] = int(support)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[name] = float(f1)
        
        if support > 0:
            f1_scores.append(f1)
            weights.append(support)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Micro F1
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    f1_macro = float(np.mean(f1_scores)) if f1_scores else 0.0
    
    # Convert CM to dict
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
        per_class_f1=per_class_f1,
        per_class_support=per_class_support,
        confusion_matrix=cm_dict,
    )


def run_inference(
    checkpoint_path: Path,
    dataset,
    device: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run inference on dataset and return predictions.
    
    Returns:
        alpha_true, alpha_pred, lengths, model_true, model_pred
    """
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model from checkpoint
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint (full experiment config)
    config = checkpoint.get("config", {})
    
    # Extract model config - could be nested under 'model' key or at top level
    model_config = config.get("model", config)
    encoder_name = model_config.get("encoder", model_config.get("encoder_name", "lstm"))
    encoder_kwargs = model_config.get("params", model_config.get("encoder_kwargs", {}))
    tasks = model_config.get("tasks", ["alpha"])
    
    # Import and create model
    from methods.predictor import DiffusionPredictor
    
    model = DiffusionPredictor(
        encoder_name=encoder_name,
        encoder_kwargs=encoder_kwargs,
        tasks=tasks,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    is_multitask = len(tasks) > 1
    
    print(f"  Loaded {encoder_name} encoder")
    print(f"  Tasks: {tasks}")
    
    alpha_true = []
    alpha_pred = []
    model_true = []
    model_pred = []
    lengths = []
    
    print(f"Running inference on {len(dataset)} trajectories...")
    
    with torch.no_grad():
        for i, traj in enumerate(dataset):
            if i % 500 == 0:
                print(f"  Processing {i}/{len(dataset)}...")
            
            # Get prediction
            positions = torch.tensor(traj.positions, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(positions)
            
            # Handle output based on tasks
            if is_multitask:
                alpha_out, model_out = output
                if alpha_out is not None:
                    alpha_pred.append(alpha_out[0].cpu().numpy())
                if model_out is not None:
                    model_pred.append(torch.argmax(model_out, dim=-1)[0].cpu().numpy())
            else:
                if 'alpha' in tasks:
                    alpha_pred.append(output[0].cpu().numpy())
                elif 'model' in tasks:
                    model_pred.append(torch.argmax(output, dim=-1)[0].cpu().numpy())
            
            alpha_true.append(traj.alpha_true)
            model_true.append(traj.model_true if traj.model_true is not None else -1)
            lengths.append(len(traj.positions))
    
    return (
        np.array(alpha_true),
        np.array(alpha_pred).flatten() if alpha_pred else None,
        np.array(lengths),
        np.array(model_true) if any(m >= 0 for m in model_true) else None,
        np.array(model_pred).flatten() if model_pred else None,
    )


def analyze_checkpoint(
    checkpoint_path: Path,
    output_dir: Optional[Path] = None,
    max_samples: int = 5000,
    device: str = "auto",
):
    """Full analysis of a single checkpoint (supports Task 1 and Task 2)."""
    
    checkpoint_path = Path(checkpoint_path)
    if output_dir is None:
        output_dir = checkpoint_path.parent / "analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = checkpoint_path.parent.name
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")
    
    # Load test data
    print("\nLoading test data...")
    dataset = load_from_hf(
        split="test",
        max_trajectories=max_samples,
        shuffle=True,
        seed=42,
    )
    
    # Run inference
    alpha_true, alpha_pred, lengths, model_true, model_pred = run_inference(
        checkpoint_path, dataset, device
    )
    
    results = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
    }
    
    # Task 1: Alpha prediction metrics
    if alpha_pred is not None:
        print("\nComputing Task 1 (α inference) metrics...")
        overall_metrics = compute_extended_metrics(alpha_true, alpha_pred)
        
        print(f"\n{'='*40}")
        print("TASK 1: α INFERENCE METRICS")
        print(f"{'='*40}")
        print(f"  Samples:        {overall_metrics.n_samples:,}")
        print(f"  MAE:            {overall_metrics.mae:.4f}")
        print(f"  RMSE:           {overall_metrics.rmse:.4f}")
        print(f"  Median AE:      {overall_metrics.median_ae:.4f}")
        print(f"  R²:             {overall_metrics.r2:.4f}")
        print(f"  Pearson r:      {overall_metrics.pearson_r:.4f}")
        print(f"  Bias:           {overall_metrics.bias:.4f} ± {overall_metrics.bias_std:.4f}")
        print(f"  P90 Error:      {overall_metrics.p90_error:.4f}")
        print(f"  P95 Error:      {overall_metrics.p95_error:.4f}")
        print(f"  Max Error:      {overall_metrics.max_error:.4f}")
        
        # Breakdown by α
        print(f"\n{'='*40}")
        print("BREAKDOWN BY α VALUE")
        print(f"{'='*40}")
        alpha_breakdown = breakdown_by_alpha(alpha_true, alpha_pred)
        for name, metrics in alpha_breakdown.items():
            print(f"  {name:25s}: MAE={metrics.mae:.4f}, n={metrics.n_samples}")
        
        # Breakdown by length
        print(f"\n{'='*40}")
        print("BREAKDOWN BY TRAJECTORY LENGTH")
        print(f"{'='*40}")
        length_breakdown = breakdown_by_length(alpha_true, alpha_pred, lengths)
        for name, metrics in length_breakdown.items():
            print(f"  {name:15s}: MAE={metrics.mae:.4f}, n={metrics.n_samples}")
        
        # Create visualizations
        print(f"\nCreating Task 1 visualizations...")
        create_visualizations(alpha_true, alpha_pred, lengths, output_dir, model_name)
        
        results["n_samples"] = int(overall_metrics.n_samples)
        results["task1_overall"] = overall_metrics.to_dict()
        results["task1_by_alpha"] = {name: m.to_dict() for name, m in alpha_breakdown.items()}
        results["task1_by_length"] = {name: m.to_dict() for name, m in length_breakdown.items()}
    
    # Task 2: Model classification metrics
    if model_pred is not None and model_true is not None:
        print("\nComputing Task 2 (model classification) metrics...")
        class_metrics = compute_classification_metrics(model_true, model_pred)
        
        print(f"\n{'='*40}")
        print("TASK 2: MODEL CLASSIFICATION METRICS")
        print(f"{'='*40}")
        print(f"  Samples:        {class_metrics.n_samples:,}")
        print(f"  Accuracy:       {class_metrics.accuracy:.4f}")
        print(f"  F1 (micro):     {class_metrics.f1_micro:.4f}  ← Primary ANDI metric")
        print(f"  F1 (macro):     {class_metrics.f1_macro:.4f}")
        
        print(f"\n  Per-class F1:")
        for name in MODEL_NAMES:
            if name in class_metrics.per_class_f1:
                f1 = class_metrics.per_class_f1[name]
                support = class_metrics.per_class_support.get(name, 0)
                print(f"    {name:<10}: F1={f1:.4f}, n={support}")
        
        results["task2_classification"] = class_metrics.to_dict()
        
        # Create confusion matrix visualization
        if class_metrics.confusion_matrix:
            print("\nCreating Task 2 visualizations...")
            create_confusion_matrix_viz(class_metrics.confusion_matrix, output_dir, model_name)
    
    # Save results to JSON
    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'analysis_results.json'}")
    
    return results


def create_confusion_matrix_viz(
    cm: Dict[str, Dict[str, int]],
    output_dir: Path,
    model_name: str,
):
    """Create confusion matrix visualization."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available, skipping confusion matrix visualization")
        return
    
    output_dir = Path(output_dir)
    class_names = MODEL_NAMES
    
    # Convert dict to numpy array
    n_classes = len(class_names)
    cm_array = np.zeros((n_classes, n_classes))
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            cm_array[i, j] = cm.get(true_name, {}).get(pred_name, 0)
    
    # Normalize for display
    cm_norm = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Model',
        xlabel='Predicted Model',
    )
    ax.set_title(f'{model_name}: Confusion Matrix (Task 2)')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f'{int(cm_array[i, j])}\n({cm_norm[i, j]:.2f})',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    print(f"  Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")


def compare_models(
    outputs_dir: Path,
    output_file: Optional[Path] = None,
    max_samples: int = 2000,
):
    """Compare all models in outputs directory."""
    
    outputs_dir = Path(outputs_dir)
    if output_file is None:
        output_file = outputs_dir / "model_comparison.json"
    
    # Find all checkpoints
    checkpoints = list(outputs_dir.glob("*/best.pt"))
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    print(f"\nFound {len(checkpoints)} models to compare")
    
    all_results = {}
    for checkpoint in sorted(checkpoints):
        model_name = checkpoint.parent.name
        try:
            results = analyze_checkpoint(
                checkpoint,
                max_samples=max_samples,
            )
            all_results[model_name] = results
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    # Create comparison summary - Task 1
    has_task1 = any("task1_overall" in r for r in all_results.values())
    has_task2 = any("task2_classification" in r for r in all_results.values())
    
    if has_task1:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY - TASK 1 (α INFERENCE)")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Bias':>10}")
        print("-" * 80)
        
        models_with_task1 = [(n, r) for n, r in all_results.items() if "task1_overall" in r]
        sorted_models = sorted(models_with_task1, key=lambda x: x[1]["task1_overall"]["mae"])
        for model_name, results in sorted_models:
            m = results["task1_overall"]
            print(f"{model_name:<20} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f} {m['bias']:>10.4f}")
    
    if has_task2:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY - TASK 2 (MODEL CLASSIFICATION)")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Accuracy':>10} {'F1 (micro)':>12} {'F1 (macro)':>12}")
        print("-" * 80)
        
        models_with_task2 = [(n, r) for n, r in all_results.items() if "task2_classification" in r]
        sorted_models = sorted(models_with_task2, key=lambda x: x[1]["task2_classification"]["f1_micro"], reverse=True)
        for model_name, results in sorted_models:
            m = results["task2_classification"]
            print(f"{model_name:<20} {m['accuracy']:>10.4f} {m['f1_micro']:>12.4f} {m['f1_macro']:>12.4f}")
    
    # Save comparison
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComparison saved to {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze model results")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--compare", type=str, help="Compare all models in directory")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples for analysis")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(
            Path(args.compare),
            max_samples=args.max_samples,
        )
    elif args.checkpoint:
        analyze_checkpoint(
            Path(args.checkpoint),
            output_dir=Path(args.output) if args.output else None,
            max_samples=args.max_samples,
            device=args.device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

