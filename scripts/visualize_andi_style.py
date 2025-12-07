#!/usr/bin/env python3
"""
Generate ANDI Challenge-style visualizations for model comparison.

Creates figures similar to those in the ANDI Challenge paper:
https://www.nature.com/articles/s41467-021-26320-w

Figures generated:
1. Method comparison bar charts (MAE for Task 1, F1 for Task 2)
2. Performance heatmaps (MAE by α range and trajectory length)
3. Confusion matrices for model classification
4. Scatter plots of predicted vs true α
5. Performance breakdown by diffusion model

Usage:
    python scripts/visualize_andi_style.py
    python scripts/visualize_andi_style.py --output outputs/andi_figures
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

from data.trajectory import MODEL_NAMES, NUM_MODELS

# ANDI paper color scheme
COLORS = {
    'lstm': '#1f77b4',
    'lstm-multitask': '#ff7f0e', 
    'cnn': '#2ca02c',
    'cnn-multitask': '#d62728',
    'hybrid': '#9467bd',
    'hybrid-multitask': '#8c564b',
    'mlp': '#e377c2',
    'mlp-multitask': '#7f7f7f',
    'msd_fitting': '#bcbd22',
}

# Model colors for confusion matrix
MODEL_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']


def load_comparison_results(outputs_dir: Path) -> Dict:
    """Load model comparison results from JSON."""
    comparison_file = outputs_dir / "model_comparison.json"
    if not comparison_file.exists():
        print(f"No comparison file found at {comparison_file}")
        print("Run: sbatch slurm/submit.slurm analyze-all")
        return {}
    
    with open(comparison_file) as f:
        return json.load(f)


def load_individual_results(outputs_dir: Path) -> Dict:
    """Load individual model analysis results."""
    results = {}
    for analysis_file in outputs_dir.glob("*/analysis/analysis_results.json"):
        model_name = analysis_file.parent.parent.name
        with open(analysis_file) as f:
            results[model_name] = json.load(f)
    return results


def create_task1_comparison(results: Dict, output_dir: Path):
    """
    Create ANDI-style bar chart comparing methods on Task 1 (α inference).
    
    Similar to Fig. 2a in the ANDI paper.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract Task 1 metrics
    models = []
    maes = []
    rmses = []
    
    for name, data in sorted(results.items()):
        if 'task1_overall' in data:
            models.append(name)
            maes.append(data['task1_overall']['mae'])
            rmses.append(data['task1_overall']['rmse'])
    
    if not models:
        print("No Task 1 results found")
        return
    
    # Sort by MAE
    sorted_idx = np.argsort(maes)
    models = [models[i] for i in sorted_idx]
    maes = [maes[i] for i in sorted_idx]
    rmses = [rmses[i] for i in sorted_idx]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Colors
    colors = [COLORS.get(m, '#333333') for m in models]
    
    bars1 = ax.bar(x - width/2, maes, width, label='MAE', color=colors, alpha=0.9)
    bars2 = ax.bar(x + width/2, rmses, width, label='RMSE', color=colors, alpha=0.5, hatch='//')
    
    # Add value labels
    for bar, val in zip(bars1, maes):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Task 1: Anomalous Exponent (α) Inference\n(Lower is better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(rmses) * 1.2)
    
    # Add ANDI reference line (approximate best from paper)
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='ANDI best ~0.15')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'task1_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'task1_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved Task 1 comparison to {output_dir / 'task1_comparison.png'}")


def create_task2_comparison(results: Dict, output_dir: Path):
    """
    Create ANDI-style bar chart comparing methods on Task 2 (model classification).
    
    Similar to Fig. 2b in the ANDI paper.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract Task 2 metrics
    models = []
    f1_micros = []
    accuracies = []
    
    for name, data in sorted(results.items()):
        if 'task2_classification' in data:
            models.append(name)
            f1_micros.append(data['task2_classification']['f1_micro'])
            accuracies.append(data['task2_classification']['accuracy'])
    
    if not models:
        print("No Task 2 results found")
        return
    
    # Sort by F1 (descending)
    sorted_idx = np.argsort(f1_micros)[::-1]
    models = [models[i] for i in sorted_idx]
    f1_micros = [f1_micros[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    
    x = np.arange(len(models))
    width = 0.35
    
    colors = [COLORS.get(m, '#333333') for m in models]
    
    bars1 = ax.bar(x - width/2, f1_micros, width, label='F1 (micro)', color=colors, alpha=0.9)
    bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color=colors, alpha=0.5, hatch='//')
    
    # Add value labels
    for bar, val in zip(bars1, f1_micros):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Task 2: Diffusion Model Classification\n(Higher is better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    # Add ANDI reference line
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='ANDI best ~0.85')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'task2_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'task2_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved Task 2 comparison to {output_dir / 'task2_comparison.png'}")


def create_alpha_breakdown_heatmap(results: Dict, output_dir: Path):
    """
    Create heatmap showing MAE breakdown by α range for each method.
    
    Similar to Fig. 3 in the ANDI paper.
    """
    # Get all models with Task 1 breakdown
    models_with_breakdown = []
    alpha_ranges = []
    
    for name, data in results.items():
        if 'task1_by_alpha' in data and data['task1_by_alpha']:
            models_with_breakdown.append(name)
            if not alpha_ranges:
                alpha_ranges = list(data['task1_by_alpha'].keys())
    
    if not models_with_breakdown or not alpha_ranges:
        print("No α breakdown data found")
        return
    
    # Build matrix
    n_models = len(models_with_breakdown)
    n_ranges = len(alpha_ranges)
    
    mae_matrix = np.zeros((n_models, n_ranges))
    
    for i, model in enumerate(models_with_breakdown):
        for j, alpha_range in enumerate(alpha_ranges):
            if alpha_range in results[model]['task1_by_alpha']:
                mae_matrix[i, j] = results[model]['task1_by_alpha'][alpha_range]['mae']
            else:
                mae_matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(n_ranges))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels([r.replace('_', '\n') for r in alpha_ranges], fontsize=10)
    ax.set_yticklabels(models_with_breakdown, fontsize=10)
    
    # Add text annotations
    for i in range(n_models):
        for j in range(n_ranges):
            if not np.isnan(mae_matrix[i, j]):
                text = ax.text(j, i, f'{mae_matrix[i, j]:.3f}',
                              ha='center', va='center', fontsize=9,
                              color='white' if mae_matrix[i, j] > 0.2 else 'black')
    
    ax.set_xlabel('α Range', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('Task 1: MAE by α Range\n(Subdiffusion → Normal → Superdiffusion)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('MAE', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_breakdown_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'alpha_breakdown_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved α breakdown heatmap to {output_dir / 'alpha_breakdown_heatmap.png'}")


def create_length_breakdown_heatmap(results: Dict, output_dir: Path):
    """
    Create heatmap showing MAE breakdown by trajectory length for each method.
    """
    models_with_breakdown = []
    length_ranges = []
    
    for name, data in results.items():
        if 'task1_by_length' in data and data['task1_by_length']:
            models_with_breakdown.append(name)
            if not length_ranges:
                length_ranges = list(data['task1_by_length'].keys())
    
    if not models_with_breakdown or not length_ranges:
        print("No length breakdown data found")
        return
    
    n_models = len(models_with_breakdown)
    n_ranges = len(length_ranges)
    
    mae_matrix = np.zeros((n_models, n_ranges))
    
    for i, model in enumerate(models_with_breakdown):
        for j, length_range in enumerate(length_ranges):
            if length_range in results[model]['task1_by_length']:
                mae_matrix[i, j] = results[model]['task1_by_length'][length_range]['mae']
            else:
                mae_matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(n_ranges))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(length_ranges, fontsize=10)
    ax.set_yticklabels(models_with_breakdown, fontsize=10)
    
    for i in range(n_models):
        for j in range(n_ranges):
            if not np.isnan(mae_matrix[i, j]):
                text = ax.text(j, i, f'{mae_matrix[i, j]:.3f}',
                              ha='center', va='center', fontsize=9,
                              color='white' if mae_matrix[i, j] > 0.2 else 'black')
    
    ax.set_xlabel('Trajectory Length', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title('Task 1: MAE by Trajectory Length\n(Short → Long)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('MAE', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'length_breakdown_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'length_breakdown_heatmap.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved length breakdown heatmap to {output_dir / 'length_breakdown_heatmap.png'}")


def create_confusion_matrix_grid(results: Dict, output_dir: Path):
    """
    Create grid of confusion matrices for Task 2 (model classification).
    
    Similar to Fig. 4 in the ANDI paper.
    """
    models_with_cm = [(name, data) for name, data in results.items() 
                      if 'task2_classification' in data and data['task2_classification'].get('confusion_matrix')]
    
    if not models_with_cm:
        print("No confusion matrices found")
        return
    
    n_models = len(models_with_cm)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, data) in enumerate(models_with_cm):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        cm_dict = data['task2_classification']['confusion_matrix']
        
        # Convert to array
        cm = np.zeros((NUM_MODELS, NUM_MODELS))
        for i, true_name in enumerate(MODEL_NAMES):
            for j, pred_name in enumerate(MODEL_NAMES):
                cm[i, j] = cm_dict.get(true_name, {}).get(pred_name, 0)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(NUM_MODELS))
        ax.set_yticks(np.arange(NUM_MODELS))
        ax.set_xticklabels(MODEL_NAMES, fontsize=9)
        ax.set_yticklabels(MODEL_NAMES, fontsize=9)
        
        # Text annotations
        for i in range(NUM_MODELS):
            for j in range(NUM_MODELS):
                text = ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if cm_norm[i, j] > 0.5 else 'black')
        
        f1 = data['task2_classification']['f1_micro']
        ax.set_title(f'{model_name}\nF1={f1:.3f}', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
    
    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Task 2: Confusion Matrices by Method', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'confusion_matrices.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved confusion matrices to {output_dir / 'confusion_matrices.png'}")


def create_per_model_f1_comparison(results: Dict, output_dir: Path):
    """
    Create grouped bar chart showing F1 score per diffusion model.
    
    Shows which diffusion models are hardest to classify.
    """
    methods_with_f1 = []
    per_model_f1 = {}
    
    for name, data in results.items():
        if 'task2_classification' in data:
            per_class = data['task2_classification'].get('per_class_f1', {})
            if per_class:
                methods_with_f1.append(name)
                per_model_f1[name] = per_class
    
    if not methods_with_f1:
        print("No per-model F1 data found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(MODEL_NAMES))
    width = 0.8 / len(methods_with_f1)
    
    for i, method in enumerate(methods_with_f1):
        f1_values = [per_model_f1[method].get(m, 0) for m in MODEL_NAMES]
        offset = (i - len(methods_with_f1)/2 + 0.5) * width
        color = COLORS.get(method, f'C{i}')
        ax.bar(x + offset, f1_values, width, label=method, color=color, alpha=0.8)
    
    ax.set_xlabel('Diffusion Model', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Task 2: F1 Score by Diffusion Model Type', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_NAMES, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_model_f1.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'per_model_f1.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved per-model F1 comparison to {output_dir / 'per_model_f1.png'}")


def create_summary_table(results: Dict, output_dir: Path):
    """Create summary table similar to ANDI paper Table 1."""
    
    summary = []
    summary.append("=" * 90)
    summary.append("ANDI-STYLE RESULTS SUMMARY")
    summary.append("=" * 90)
    summary.append("")
    
    # Task 1 table
    summary.append("TASK 1: Anomalous Exponent (α) Inference")
    summary.append("-" * 90)
    summary.append(f"{'Method':<20} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Bias':>10} {'P90':>10}")
    summary.append("-" * 90)
    
    task1_results = [(n, d) for n, d in results.items() if 'task1_overall' in d]
    task1_results.sort(key=lambda x: x[1]['task1_overall']['mae'])
    
    for name, data in task1_results:
        m = data['task1_overall']
        summary.append(f"{name:<20} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f} {m['bias']:>10.4f} {m['p90_error']:>10.4f}")
    
    summary.append("")
    
    # Task 2 table
    summary.append("TASK 2: Diffusion Model Classification")
    summary.append("-" * 90)
    summary.append(f"{'Method':<20} {'F1 (micro)':>12} {'F1 (macro)':>12} {'Accuracy':>12}")
    summary.append("-" * 90)
    
    task2_results = [(n, d) for n, d in results.items() if 'task2_classification' in d]
    task2_results.sort(key=lambda x: x[1]['task2_classification']['f1_micro'], reverse=True)
    
    for name, data in task2_results:
        m = data['task2_classification']
        summary.append(f"{name:<20} {m['f1_micro']:>12.4f} {m['f1_macro']:>12.4f} {m['accuracy']:>12.4f}")
    
    summary.append("")
    summary.append("=" * 90)
    summary.append("ANDI Challenge Reference (2D trajectories):")
    summary.append("  Task 1 best MAE: ~0.15-0.20")
    summary.append("  Task 2 best F1:  ~0.80-0.85")
    summary.append("=" * 90)
    
    # Print and save
    summary_text = "\n".join(summary)
    print(summary_text)
    
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"\n  Saved summary to {output_dir / 'summary.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Generate ANDI-style visualizations")
    parser.add_argument("--outputs", type=str, default="outputs", help="Outputs directory")
    parser.add_argument("--output", type=str, default="outputs/andi_figures", help="Output directory for figures")
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib required for visualizations")
        sys.exit(1)
    
    outputs_dir = Path(args.outputs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ANDI Challenge-Style Visualization")
    print("=" * 60)
    print(f"Loading results from: {outputs_dir}")
    print(f"Saving figures to: {output_dir}")
    print()
    
    # Load results
    results = load_comparison_results(outputs_dir)
    if not results:
        # Try loading individual results
        results = load_individual_results(outputs_dir)
    
    if not results:
        print("No results found. Run analysis first:")
        print("  sbatch slurm/submit.slurm analyze-all")
        sys.exit(1)
    
    print(f"Found results for {len(results)} models: {list(results.keys())}")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    
    create_task1_comparison(results, output_dir)
    create_task2_comparison(results, output_dir)
    create_alpha_breakdown_heatmap(results, output_dir)
    create_length_breakdown_heatmap(results, output_dir)
    create_confusion_matrix_grid(results, output_dir)
    create_per_model_f1_comparison(results, output_dir)
    create_summary_table(results, output_dir)
    
    print()
    print("=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()


