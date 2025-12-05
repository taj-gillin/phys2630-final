#!/usr/bin/env python3
"""
Run method comparison experiment.

Usage:
    python scripts/run_comparison.py --config experiments/exp_comparison.yaml
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.synthetic import generate_dataset
from data.hf_loader import load_from_hf
from methods import MSDFitting, MSDPINN, DisplacementPINN
from evaluation import compare_methods, save_results, print_comparison_table
from utils.config import load_config, ensure_dir


def load_dataset(cfg: dict):
    """Load dataset from HuggingFace or generate synthetically."""
    source = cfg["data"].get("source", "synthetic")
    
    if source == "huggingface":
        print("\n[1/3] Loading trajectories from HuggingFace...")
        dataset = load_from_hf(
            repo_id=cfg["data"].get("repo_id", "taj-gillin/andi-trajectory"),
            alphas=cfg["data"].get("alphas"),
            lengths=cfg["data"].get("lengths"),
            max_trajectories=cfg["data"].get("max_trajectories"),
        )
        print(f"  Loaded {len(dataset)} trajectories from HuggingFace")
    else:
        print("\n[1/3] Generating synthetic FBM trajectories...")
        dataset = generate_dataset(
            alphas=cfg["data"]["alphas"],
            trajectories_per_alpha=cfg["data"]["trajectories_per_alpha"],
            trajectory_length=cfg["data"]["trajectory_length"],
            D0=cfg["data"]["D0"],
            seed=cfg["data"].get("seed"),
        )
        print(f"  Generated {len(dataset)} trajectories")
        print(f"  Alpha values: {cfg['data']['alphas']}")
        print(f"  Trajectory length: {cfg['data']['trajectory_length']}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Run method comparison experiment")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    print("=" * 60)
    print("Per-Trajectory PINN Comparison Experiment")
    print("=" * 60)
    
    # Load dataset (from HF or generate)
    dataset = load_dataset(cfg)
    
    # Initialize methods
    print("\n[2/3] Initializing inference methods...")
    methods = []
    
    # MSD Fitting baseline
    msd_cfg = cfg["methods"]["msd_fitting"]
    methods.append(MSDFitting(
        max_lag_fraction=msd_cfg["max_lag_fraction"],
    ))
    print(f"  - MSD Fitting (baseline)")
    
    # MSD PINN
    pinn_cfg = cfg["methods"]["msd_pinn"]
    methods.append(MSDPINN(
        epochs=pinn_cfg["epochs"],
        lr=pinn_cfg["lr"],
        alpha_init=pinn_cfg.get("alpha_init", 0.8),
        D0_init=pinn_cfg.get("D0_init", 1.0),
        max_lag_fraction=pinn_cfg.get("max_lag_fraction", 0.25),
    ))
    print(f"  - MSD PINN ({pinn_cfg['epochs']} epochs)")
    
    # Displacement PINN
    disp_cfg = cfg["methods"]["displacement_pinn"]
    methods.append(DisplacementPINN(
        hidden_layers=disp_cfg["hidden_layers"],
        hidden_dim=disp_cfg["hidden_dim"],
        epochs=disp_cfg["epochs"],
        lr=disp_cfg["lr"],
        lambda_physics=disp_cfg["lambda_physics"],
        alpha_init=disp_cfg.get("alpha_init", 0.8),
        D0_init=disp_cfg.get("D0_init", 1.0),
        max_lag_fraction=disp_cfg.get("max_lag_fraction", 0.25),
        n_collocation=disp_cfg.get("n_collocation", 500),
    ))
    print(f"  - Displacement PINN ({disp_cfg['epochs']} epochs, {disp_cfg['hidden_layers']}x{disp_cfg['hidden_dim']} network)")
    
    # Run comparison
    print("\n[3/3] Running comparison...")
    comparison = compare_methods(methods, dataset, verbose=True)
    
    # Print summary table
    print_comparison_table(comparison["summaries"])
    
    # Save results
    output_dir = Path(cfg["output"]["dir"])
    ensure_dir(output_dir)
    save_results(comparison, output_dir)
    print(f"\nResults saved to: {output_dir}/comparison_results.json")
    
    # Print per-alpha breakdown
    print("\n" + "=" * 60)
    print("Per-Alpha Breakdown")
    print("=" * 60)
    
    for alpha in cfg["data"]["alphas"]:
        print(f"\nAlpha = {alpha}:")
        for method_name, results in comparison["results"].items():
            alpha_results = [r for r in results if abs(r.alpha_true - alpha) < 0.01]
            if alpha_results:
                errors = [r.alpha_error for r in alpha_results if not __import__('numpy').isnan(r.alpha_pred)]
                if errors:
                    mean_err = __import__('numpy').mean(errors)
                    print(f"  {method_name:<25}: error = {mean_err:.4f}")
    
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

