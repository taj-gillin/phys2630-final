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
from methods import MSDFitting, DiffusionPredictorInference
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
        print(f"  Loaded {len(dataset)} trajectories")
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
    print(f"Checkpoints: {model_dir}")
    print(f"Start: {datetime.now()}")
    
    dataset = load_dataset(cfg)
    
    print("\n[2/4] Loading models...")
    methods = []
    
    # Baseline
    methods.append(MSDFitting(max_lag_fraction=0.25))
    print("  ✓ MSD Fitting (baseline)")
    
    # Load each trained model
    model_names = ["linear", "mlp", "lstm", "cnn", "hybrid", "hybrid_pinn"]
    
    for name in model_names:
        if not cfg["models"].get(name, {}).get("enabled", True):
            continue
            
        checkpoint_path = model_dir / f"{name}_best.pt"
        if checkpoint_path.exists():
            try:
                inference = DiffusionPredictorInference(
                    checkpoint_path=str(checkpoint_path),
                    encoder_name=name.replace("_pinn", ""),
                    device=device,
                )
                # Override name to be more descriptive
                inference.name = name.upper().replace("_", " + ")
                methods.append(inference)
                print(f"  ✓ {inference.name}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        else:
            print(f"  ✗ {name}: checkpoint not found")
    
    if len(methods) < 2:
        print("\nError: Need at least 2 models. Train models first:")
        print("  python scripts/train_models.py --config experiments/exp_train_models.yaml")
        return
    
    print("\n[3/4] Running comparison...")
    comparison = compare_methods(methods, dataset, verbose=True)
    
    print_comparison_table(comparison["summaries"])
    
    output_dir = Path(cfg["output"]["dir"])
    ensure_dir(output_dir)
    save_results(comparison, output_dir)
    print(f"\nResults saved to: {output_dir}")
    
    # Per-alpha breakdown
    print("\n[4/4] Per-Alpha Analysis")
    print("=" * 60)
    
    alphas = cfg["data"].get("alphas", [0.3, 0.5, 0.7, 0.9, 1.0])
    
    for alpha in alphas:
        print(f"\nα = {alpha}:")
        for method_name, results in comparison["results"].items():
            alpha_results = [r for r in results if abs(r.alpha_true - alpha) < 0.01]
            if alpha_results:
                errors = [r.alpha_error for r in alpha_results if not np.isnan(r.alpha_pred)]
                if errors:
                    print(f"  {method_name:<20}: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
    
    print("\n" + "=" * 60)
    print(f"Complete! End: {datetime.now()}")


if __name__ == "__main__":
    main()
