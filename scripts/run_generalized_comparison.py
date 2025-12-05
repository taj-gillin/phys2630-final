#!/usr/bin/env python3
"""
Compare pre-trained generalized models against baselines.

Usage:
    python scripts/run_generalized_comparison.py --config experiments/exp_generalized_comparison.yaml
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.synthetic import generate_dataset
from data.hf_loader import load_from_hf
from methods import (
    MSDFitting,
    LSTMInference,
    CNNInference,
    TrajectoryPINNInference,
)
from evaluation import compare_methods, save_results, print_comparison_table
from utils.config import load_config, ensure_dir


def load_dataset(cfg: dict):
    """Load dataset from HuggingFace or generate synthetically."""
    source = cfg["data"].get("source", "synthetic")
    
    if source == "huggingface":
        print("\n[1/4] Loading trajectories from HuggingFace...")
        dataset = load_from_hf(
            repo_id=cfg["data"].get("repo_id", "taj-gillin/andi-trajectory"),
            alphas=cfg["data"].get("alphas"),
            lengths=cfg["data"].get("lengths"),
            max_trajectories=cfg["data"].get("max_trajectories"),
        )
        print(f"  Loaded {len(dataset)} trajectories from HuggingFace")
    else:
        print("\n[1/4] Generating synthetic test trajectories...")
        dataset = generate_dataset(
            alphas=cfg["data"]["alphas"],
            trajectories_per_alpha=cfg["data"]["trajectories_per_alpha"],
            trajectory_length=cfg["data"]["trajectory_length"],
            D0=cfg["data"]["D0"],
            seed=cfg["data"].get("seed"),
        )
        print(f"  Generated {len(dataset)} trajectories")
        print(f"  Alpha values: {cfg['data']['alphas']}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Compare generalized models")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(cfg["models"]["checkpoint_dir"])
    
    print("=" * 60)
    print("Generalized Model Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model checkpoints: {model_dir}")
    print(f"Start time: {datetime.now()}")
    
    # Load dataset
    dataset = load_dataset(cfg)
    
    # Initialize methods
    print("\n[2/4] Loading pre-trained models...")
    methods = []
    
    # Baseline: MSD Fitting
    methods.append(MSDFitting(max_lag_fraction=0.25))
    print("  ✓ MSD Fitting (baseline)")
    
    # LSTM
    if cfg["models"].get("lstm", {}).get("enabled", True):
        lstm_path = model_dir / "lstm_best.pt"
        if lstm_path.exists():
            methods.append(LSTMInference(checkpoint_path=str(lstm_path), device=device))
            print(f"  ✓ LSTM Predictor (loaded from {lstm_path})")
        else:
            print(f"  ✗ LSTM checkpoint not found: {lstm_path}")
    
    # CNN
    if cfg["models"].get("cnn", {}).get("enabled", True):
        cnn_path = model_dir / "cnn_best.pt"
        if cnn_path.exists():
            methods.append(CNNInference(checkpoint_path=str(cnn_path), device=device))
            print(f"  ✓ CNN Predictor (loaded from {cnn_path})")
        else:
            print(f"  ✗ CNN checkpoint not found: {cnn_path}")
    
    # Trajectory PINN
    if cfg["models"].get("pinn", {}).get("enabled", True):
        pinn_path = model_dir / "pinn_best.pt"
        if pinn_path.exists():
            methods.append(TrajectoryPINNInference(checkpoint_path=str(pinn_path), device=device))
            print(f"  ✓ Trajectory PINN (loaded from {pinn_path})")
        else:
            print(f"  ✗ PINN checkpoint not found: {pinn_path}")
    
    if len(methods) < 2:
        print("\nError: Not enough models loaded. Please train models first.")
        print("Run: python scripts/train_models.py --config experiments/exp_train_models.yaml")
        return
    
    # Run comparison
    print("\n[3/4] Running comparison...")
    comparison = compare_methods(methods, dataset, verbose=True)
    
    # Print summary table
    print_comparison_table(comparison["summaries"])
    
    # Save results
    output_dir = Path(cfg["output"]["dir"])
    ensure_dir(output_dir)
    save_results(comparison, output_dir)
    print(f"\nResults saved to: {output_dir}/comparison_results.json")
    
    # Per-alpha breakdown
    print("\n[4/4] Per-Alpha Analysis")
    print("=" * 60)
    
    alphas = cfg["data"].get("alphas", [0.3, 0.5, 0.7, 0.9, 1.0])
    
    for alpha in alphas:
        print(f"\nAlpha = {alpha}:")
        for method_name, results in comparison["results"].items():
            alpha_results = [r for r in results if abs(r.alpha_true - alpha) < 0.01]
            if alpha_results:
                errors = [r.alpha_error for r in alpha_results if not np.isnan(r.alpha_pred)]
                if errors:
                    mean_err = np.mean(errors)
                    std_err = np.std(errors)
                    print(f"  {method_name:<25}: error = {mean_err:.4f} ± {std_err:.4f}")
    
    # Compute speedup
    print("\n" + "=" * 60)
    print("Inference Time Comparison")
    print("=" * 60)
    
    # Note: The generalized models should be MUCH faster since they don't train
    # This is a key advantage over per-trajectory methods
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()

