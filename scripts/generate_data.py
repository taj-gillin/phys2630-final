#!/usr/bin/env python3
"""
Generate FBM trajectories and upload to HuggingFace Hub.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w),
trajectories are normalized with MSD(τ) ∝ τ^α. We focus on predicting α only.

Usage:
    python scripts/generate_data.py --config configs/data/generate.yaml
    sbatch slurm/submit.slurm generate
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from utils.wandb_logger import init_wandb


def generate_trajectory(traj_id: int, seed: int, alpha_range: tuple, length_range: tuple) -> dict:
    """
    Generate a single 2D FBM trajectory following AnDi Challenge conventions.
    
    AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w):
        - Task 1: Infer α (anomalous exponent)
        - Task 2: Classify diffusion model
        - Trajectories are normalized with MSD(τ) ∝ τ^α
        
    We use andi_datasets native 2D generation which produces normalized 
    trajectories where Var[x(T)] ≈ 1, giving MSD(τ) ≈ d·(τ/T)^α for d dimensions.
    """
    rng = np.random.RandomState(seed + traj_id)
    
    alpha = rng.uniform(*alpha_range)
    length = rng.randint(length_range[0], length_range[1] + 1)
    
    from andi_datasets.models_theory import models_theory
    MT = models_theory()
    
    np.random.seed(seed + traj_id)
    
    # Use native 2D generation: D=2 returns [x₁...xₜ, y₁...yₜ]
    traj_2d = MT.fbm(T=length, alpha=alpha, D=2)
    traj_x = traj_2d[:length]
    traj_y = traj_2d[length:]
    
    # No scaling needed - use normalized trajectories as per AnDi convention
    # MSD(τ) ∝ (τ/T)^α, which preserves the α we want to infer
    
    return {
        "trajectory_id": traj_id,
        "alpha": float(alpha),
        "length": int(length),
        "x": traj_x.tolist(),
        "y": traj_y.tolist(),
    }


def generate_batch(start_idx: int, batch_size: int, seed: int, alpha_range: tuple, length_range: tuple) -> list:
    """Generate a batch of trajectories."""
    return [
        generate_trajectory(start_idx + i, seed, alpha_range, length_range)
        for i in range(batch_size)
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate and upload trajectory dataset")
    parser.add_argument("--config", default="configs/data/generate.yaml", help="Path to config")
    parser.add_argument("--dry-run", action="store_true", help="Generate but don't upload")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    hf_cfg = config["huggingface"]
    gen_cfg = config["generation"]
    proc_cfg = config.get("processing", {})
    splits_cfg = gen_cfg.get("splits", {})
    
    repo_id = hf_cfg["repo_id"]
    n_trajectories = gen_cfg["n_trajectories"]
    alpha_range = tuple(gen_cfg["alpha_range"])
    length_range = tuple(gen_cfg["length_range"])
    seed = gen_cfg["seed"]
    
    n_workers = proc_cfg.get("n_workers", 8)
    batch_size = proc_cfg.get("batch_size", 1000)
    
    # Initialize wandb
    logger = init_wandb(config, job_type="generate")
    
    print("=" * 60)
    print("FBM Trajectory Dataset Generation (AnDi Convention)")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Trajectories: {n_trajectories:,}")
    print(f"Alpha range: {alpha_range}")
    print(f"Length range: {length_range}")
    print(f"Workers: {n_workers}")
    if logger.run_url:
        print(f"Wandb: {logger.run_url}")
    print()
    
    # Generate trajectories
    records = []
    n_batches = (n_trajectories + batch_size - 1) // batch_size
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            actual_size = min(batch_size, n_trajectories - start_idx)
            future = executor.submit(
                generate_batch, start_idx, actual_size, seed,
                alpha_range, length_range
            )
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            records.extend(future.result())
    
    records.sort(key=lambda x: x["trajectory_id"])
    
    # Statistics
    alphas = [r["alpha"] for r in records]
    lengths = [r["length"] for r in records]
    
    print(f"\nGenerated {len(records):,} trajectories")
    print(f"Alpha:  [{min(alphas):.3f}, {max(alphas):.3f}], mean={np.mean(alphas):.3f}")
    print(f"Length: [{min(lengths)}, {max(lengths)}], mean={np.mean(lengths):.1f}")
    
    # Log to wandb
    logger.log({
        "data/n_trajectories": len(records),
        "data/alpha_min": min(alphas),
        "data/alpha_max": max(alphas),
    })
    
    if args.dry_run:
        print("\n[DRY RUN] Skipping upload")
        logger.finish()
        return
    
    # Get HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except:
            pass
    
    if not token:
        print("\nError: HF token required. Set HF_TOKEN or run 'huggingface-cli login'")
        logger.finish()
        sys.exit(1)
    
    # Create split ratios
    val_fraction = float(splits_cfg.get("val", splits_cfg.get("validation", 0.0)))
    test_fraction = float(splits_cfg.get("test", 0.0))
    train_fraction = splits_cfg.get("train")
    if train_fraction is None:
        train_fraction = 1.0 - val_fraction - test_fraction
    train_fraction = float(train_fraction)
    
    total_fraction = train_fraction + val_fraction + test_fraction
    if total_fraction > 1.0 + 1e-6:
        logger.finish()
        raise ValueError(
            f"Split fractions sum to {total_fraction:.3f} (>1). "
            "Please adjust generation.splits."
        )
    if train_fraction <= 0:
        logger.finish()
        raise ValueError("Train split must be positive.")
    
    # Shuffle once with a deterministic seed, then slice
    dataset = Dataset.from_list(records).shuffle(seed=seed)
    n_total = len(dataset)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    n_test = n_total - n_train - n_val
    
    if n_train <= 0 or n_val < 0 or n_test < 0:
        logger.finish()
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test} (total={n_total})"
        )
    
    train_ds = dataset.select(range(0, n_train))
    val_ds = dataset.select(range(n_train, n_train + n_val)) if n_val > 0 else None
    test_ds = dataset.select(range(n_train + n_val, n_total)) if n_test > 0 else None
    
    print("\nUpload splits")
    print(f"  Train: {len(train_ds):,}")
    if val_ds is not None:
        print(f"  Val:   {len(val_ds):,}")
    if test_ds is not None:
        print(f"  Test:  {len(test_ds):,}")
    
    split_dict = {"train": train_ds}
    if val_ds is not None:
        split_dict["validation"] = val_ds
    if test_ds is not None:
        split_dict["test"] = test_ds
    
    dataset = DatasetDict(split_dict)
    
    # Create and upload dataset
    print(f"\nUploading to {repo_id}...")
    dataset.push_to_hub(
        repo_id,
        token=token,
        commit_message=(
            f"Upload FBM dataset with splits "
            f"train={len(train_ds):,}, val={len(val_ds) if val_ds else 0:,}, "
            f"test={len(test_ds) if test_ds else 0:,}"
        ),
    )
    
    logger.set_summary("upload_success", True)
    logger.finish()
    
    print("\n" + "=" * 60)
    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
