#!/usr/bin/env python3
"""
Generate comparison visualizations from model_comparison.json.

Creates publication-ready plots comparing all analyzed models.

Usage:
    python scripts/visualize_comparison.py
    python scripts/visualize_comparison.py --input outputs/model_comparison.json
    sbatch slurm/submit.slurm visualize
"""

import argparse
import json
from pathlib import Path
import numpy as np

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")


# Color palette for models
MODEL_COLORS = {
    'lstm': '#2ecc71',      # Green
    'cnn': '#3498db',       # Blue
    'cnn_pinn': '#9b59b6',  # Purple
    'hybrid': '#e74c3c',    # Red
    'hybrid_pinn': '#e67e22', # Orange
    'mlp': '#1abc9c',       # Teal
    'linear': '#95a5a6',    # Gray
}

def get_color(model_name: str) -> str:
    """Get color for model, with fallback."""
    return MODEL_COLORS.get(model_name, '#34495e')


def load_comparison(input_path: Path) -> dict:
    """Load model comparison results."""
    with open(input_path) as f:
        return json.load(f)


def plot_overall_comparison(data: dict, output_dir: Path):
    """Bar chart comparing overall metrics across models."""
    
    models = list(data.keys())
    metrics = ['mae', 'rmse', 'bias', 'r2']
    metric_labels = ['MAE ↓', 'RMSE ↓', 'Bias (→0)', 'R² ↑']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [data[m]['overall'][metric] for m in models]
        colors = [get_color(m) for m in models]
        
        bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # Add reference line for bias
        if metric == 'bias':
            ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.suptitle('Model Comparison: Overall Metrics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_overall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_overall.png")


def plot_alpha_breakdown(data: dict, output_dir: Path):
    """Grouped bar chart comparing MAE across α ranges."""
    
    models = list(data.keys())
    
    # Get all alpha categories (use first model as reference)
    first_model = list(data.values())[0]
    alpha_cats = list(first_model.get('by_alpha', {}).keys())
    
    if not alpha_cats:
        print("  Skipping alpha breakdown (no data)")
        return
    
    # Prepare data
    x = np.arange(len(alpha_cats))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, model in enumerate(models):
        model_data = data[model].get('by_alpha', {})
        maes = [model_data.get(cat, {}).get('mae', 0) for cat in alpha_cats]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, maes, width, label=model.upper(), 
                     color=get_color(model), edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('α Range', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Model Performance by Diffusion Regime', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.replace('_', '\n') for cat in alpha_cats], fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_by_alpha.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_by_alpha.png")


def plot_length_breakdown(data: dict, output_dir: Path):
    """Line plot comparing MAE vs trajectory length."""
    
    models = list(data.keys())
    
    # Get length categories
    first_model = list(data.values())[0]
    length_cats = list(first_model.get('by_length', {}).keys())
    
    if not length_cats:
        print("  Skipping length breakdown (no data)")
        return
    
    # Order by typical length
    length_order = ['short', 'medium', 'long', 'very_long']
    length_cats = [c for c in length_order if c in length_cats]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in models:
        model_data = data[model].get('by_length', {})
        maes = [model_data.get(cat, {}).get('mae', np.nan) for cat in length_cats]
        ax.plot(length_cats, maes, 'o-', label=model.upper(), 
               color=get_color(model), linewidth=2, markersize=8)
    
    ax.set_xlabel('Trajectory Length Category', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Model Performance vs Trajectory Length', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    # Add expected trend annotation
    ax.annotate('Longer → Better', xy=(0.75, 0.85), xycoords='axes fraction',
               fontsize=10, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_by_length.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_by_length.png")


def plot_bias_comparison(data: dict, output_dir: Path):
    """Compare bias across models and α ranges."""
    
    models = list(data.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Overall bias with error bars
    ax = axes[0]
    biases = [data[m]['overall']['bias'] for m in models]
    bias_stds = [data[m]['overall']['bias_std'] for m in models]
    colors = [get_color(m) for m in models]
    
    bars = ax.bar(models, biases, yerr=bias_stds, color=colors, 
                  edgecolor='white', linewidth=1.5, capsize=5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero bias')
    ax.set_ylabel('Bias (α_pred - α_true)', fontsize=12)
    ax.set_title('Overall Prediction Bias', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # Right: Bias by α range (heatmap-style)
    ax = axes[1]
    
    first_model = list(data.values())[0]
    alpha_cats = list(first_model.get('by_alpha', {}).keys())
    
    if alpha_cats:
        bias_matrix = []
        for model in models:
            model_data = data[model].get('by_alpha', {})
            row = [model_data.get(cat, {}).get('bias', 0) for cat in alpha_cats]
            bias_matrix.append(row)
        
        bias_matrix = np.array(bias_matrix)
        
        im = ax.imshow(bias_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-0.2, vmax=0.2)
        
        ax.set_xticks(np.arange(len(alpha_cats)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([c.replace('_', '\n') for c in alpha_cats], fontsize=9)
        ax.set_yticklabels([m.upper() for m in models])
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(alpha_cats)):
                val = bias_matrix[i, j]
                color = 'white' if abs(val) > 0.1 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                       color=color, fontsize=9)
        
        ax.set_title('Bias by α Range', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Bias')
    
    plt.suptitle('Prediction Bias Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_bias.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_bias.png")


def plot_percentile_comparison(data: dict, output_dir: Path):
    """Compare error percentiles across models."""
    
    models = list(data.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    maes = [data[m]['overall']['mae'] for m in models]
    p90s = [data[m]['overall']['p90_error'] for m in models]
    p95s = [data[m]['overall']['p95_error'] for m in models]
    
    bars1 = ax.bar(x - width, maes, width, label='MAE', color='#3498db')
    bars2 = ax.bar(x, p90s, width, label='P90', color='#e74c3c')
    bars3 = ax.bar(x + width, p95s, width, label='P95', color='#9b59b6')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error Distribution: MAE vs Percentiles', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_percentiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_percentiles.png")


def plot_summary_table(data: dict, output_dir: Path):
    """Create a summary table as an image."""
    
    models = list(data.keys())
    
    # Prepare table data
    headers = ['Model', 'MAE', 'RMSE', 'Bias', 'R²', 'P90', 'P95']
    rows = []
    
    for model in models:
        m = data[model]['overall']
        rows.append([
            model.upper(),
            f"{m['mae']:.4f}",
            f"{m['rmse']:.4f}",
            f"{m['bias']:.4f}",
            f"{m['r2']:.4f}",
            f"{m['p90_error']:.4f}",
            f"{m['p95_error']:.4f}",
        ])
    
    # Sort by MAE
    rows.sort(key=lambda x: float(x[1]))
    
    fig, ax = plt.subplots(figsize=(12, 2 + len(models) * 0.5))
    ax.axis('off')
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#3498db'] * len(headers),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color header text white
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Highlight best MAE (first row after sorting)
    for i in range(len(headers)):
        table[(1, i)].set_facecolor('#d5f5e3')
    
    plt.title('Model Comparison Summary (Sorted by MAE)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison_table.png")


def create_all_visualizations(input_path: Path, output_dir: Path):
    """Generate all comparison visualizations."""
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualizations")
        return
    
    print(f"\nLoading comparison data from {input_path}...")
    data = load_comparison(input_path)
    
    print(f"Found {len(data)} models: {', '.join(data.keys())}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating visualizations in {output_dir}...")
    
    # Generate all plots
    plot_overall_comparison(data, output_dir)
    plot_alpha_breakdown(data, output_dir)
    plot_length_breakdown(data, output_dir)
    plot_bias_comparison(data, output_dir)
    plot_percentile_comparison(data, output_dir)
    plot_summary_table(data, output_dir)
    
    print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison visualizations")
    parser.add_argument("--input", type=str, default="outputs/model_comparison.json",
                       help="Path to model_comparison.json")
    parser.add_argument("--output", type=str, default="outputs/analysis",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        print("Run 'sbatch slurm/submit.slurm analyze-all' first")
        return
    
    create_all_visualizations(input_path, output_dir)


if __name__ == "__main__":
    main()

