#!/usr/bin/env python3
"""
Compare trained models and traditional methods.

Usage:
    python scripts/run_comparison.py --config configs/compare.yaml
    sbatch slurm/submit.slurm compare configs/compare.yaml
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.hf_loader import load_from_hf
from methods import MSDFitting, DiffusionPredictorInference
from evaluation import compare_methods, save_results, print_comparison_table
from utils.config import load_config, ensure_dir
from utils.wandb_logger import init_wandb


def get_device(config: dict) -> str:
    """Get device from config."""
    device_cfg = config.get("env", {}).get("device", "auto")
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def main():
    parser = argparse.ArgumentParser(description="Compare methods")
    parser.add_argument("--config", required=True, help="Path to config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    exp_name = config["experiment"]["name"]
    device = get_device(config)
    
    # Output directory
    output_dir = Path("outputs") / exp_name
    ensure_dir(output_dir)
    
    # Initialize wandb
    logger = init_wandb(config, job_type="compare")
    
    print("=" * 60)
    print(f"Method Comparison: {exp_name}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Start: {datetime.now()}")
    if logger.run_url:
        print(f"Wandb: {logger.run_url}")
    
    # Load test data
    data_cfg = config["data"]
    print("\nLoading test data from HuggingFace...")
    
    dataset = load_from_hf(
        repo_id=data_cfg["repo_id"],
        alphas=data_cfg.get("alphas"),
        max_trajectories=data_cfg.get("max_trajectories"),
        seed=data_cfg.get("seed", 99),
    )
    print(f"  Loaded {len(dataset)} test trajectories")
    
    # Initialize methods
    print("\nInitializing methods...")
    methods = []
    
    # Traditional methods
    methods_cfg = config.get("methods", {})
    if methods_cfg.get("msd_fitting", {}).get("enabled", True):
        cfg = methods_cfg["msd_fitting"]
        methods.append(MSDFitting(max_lag_fraction=cfg.get("max_lag_fraction", 0.25)))
        print("  ✓ MSD Fitting (baseline)")
    
    # Trained models
    models_cfg = config.get("models", {})
    for name, model_cfg in models_cfg.items():
        if not model_cfg.get("enabled", True):
            continue
        
        checkpoint = Path(model_cfg.get("checkpoint", f"outputs/{name}/best.pt"))
        if not checkpoint.exists():
            print(f"  ✗ {name}: checkpoint not found at {checkpoint}")
            continue
        
        try:
            encoder_name = name.replace("_pinn", "")
            inference = DiffusionPredictorInference(
                checkpoint_path=str(checkpoint),
                encoder_name=encoder_name,
                device=device,
            )
            inference.name = name.upper().replace("_", "+")
            methods.append(inference)
            print(f"  ✓ {inference.name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    if len(methods) < 2:
        print("\nError: Need at least 2 methods. Train models first.")
        logger.finish()
        return
    
    # Run comparison
    print("\nRunning comparison...")
    comparison = compare_methods(methods, dataset, verbose=True)
    
    # Log to wandb
    for method_name, summary in comparison["summaries"].items():
        logger.log({
            f"compare/{method_name}/alpha_mae": summary.get("alpha_mae", 0),
            f"compare/{method_name}/alpha_rmse": summary.get("alpha_rmse", 0),
            f"compare/{method_name}/time_mean": summary.get("time_mean", 0),
        })
    
    print_comparison_table(comparison["summaries"])
    
    # Save results
    save_results(comparison, output_dir)
    print(f"\nResults saved to: {output_dir}")
    
    # Per-alpha breakdown
    alphas = data_cfg.get("alphas", [])
    if alphas:
        print("\n" + "=" * 60)
        print("Per-Alpha Breakdown")
        print("=" * 60)
        
        for alpha in alphas:
            print(f"\nα = {alpha}:")
            for method_name, results in comparison["results"].items():
                alpha_results = [r for r in results if abs(r.alpha_true - alpha) < 0.05]
                if alpha_results:
                    errors = [r.alpha_error for r in alpha_results if not np.isnan(r.alpha_pred)]
                    if errors:
                        mean_err = np.mean(errors)
                        std_err = np.std(errors)
                        print(f"  {method_name:<15}: {mean_err:.4f} ± {std_err:.4f}")
                        
                        logger.log({
                            f"per_alpha/{method_name}/alpha_{alpha:.1f}": mean_err,
                        })
    
    logger.finish()
    
    print("\n" + "=" * 60)
    print(f"Complete! End: {datetime.now()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
