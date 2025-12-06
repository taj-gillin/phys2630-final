#!/usr/bin/env python3
"""
Generate FBM trajectories and upload to HuggingFace Hub.

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


def generate_trajectory(traj_id: int, seed: int, alpha_range: tuple, D0_range: tuple, length_range: tuple) -> dict:
    """Generate a single 2D FBM trajectory."""
    rng = np.random.RandomState(seed + traj_id)
    
    alpha = rng.uniform(*alpha_range)
    D0 = rng.uniform(*D0_range)
    length = rng.randint(length_range[0], length_range[1] + 1)
    
    from andi_datasets.models_theory import models_theory
    MT = models_theory()
    
    np.random.seed(seed + traj_id)
    traj_x = MT.fbm(T=length, alpha=alpha)
    traj_y = MT.fbm(T=length, alpha=alpha)
    
    scale = np.sqrt(D0)
    
    return {
        "trajectory_id": traj_id,
        "alpha": float(alpha),
        "D0": float(D0),
        "length": int(length),
        "x": (traj_x * scale).tolist(),
        "y": (traj_y * scale).tolist(),
    }


def generate_batch(start_idx: int, batch_size: int, seed: int, alpha_range: tuple, D0_range: tuple, length_range: tuple) -> list:
    """Generate a batch of trajectories."""
    return [
        generate_trajectory(start_idx + i, seed, alpha_range, D0_range, length_range)
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
    
    repo_id = hf_cfg["repo_id"]
    n_trajectories = gen_cfg["n_trajectories"]
    alpha_range = tuple(gen_cfg["alpha_range"])
    D0_range = tuple(gen_cfg["D0_range"])
    length_range = tuple(gen_cfg["length_range"])
    seed = gen_cfg["seed"]
    
    n_workers = proc_cfg.get("n_workers", 8)
    batch_size = proc_cfg.get("batch_size", 1000)
    
    # Initialize wandb
    logger = init_wandb(config, job_type="generate")
    
    print("=" * 60)
    print("FBM Trajectory Dataset Generation")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Trajectories: {n_trajectories:,}")
    print(f"Alpha range: {alpha_range}")
    print(f"D0 range: {D0_range}")
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
                alpha_range, D0_range, length_range
            )
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            records.extend(future.result())
    
    records.sort(key=lambda x: x["trajectory_id"])
    
    # Statistics
    alphas = [r["alpha"] for r in records]
    D0s = [r["D0"] for r in records]
    lengths = [r["length"] for r in records]
    
    print(f"\nGenerated {len(records):,} trajectories")
    print(f"Alpha:  [{min(alphas):.3f}, {max(alphas):.3f}], mean={np.mean(alphas):.3f}")
    print(f"D0:     [{min(D0s):.3f}, {max(D0s):.3f}], mean={np.mean(D0s):.3f}")
    print(f"Length: [{min(lengths)}, {max(lengths)}], mean={np.mean(lengths):.1f}")
    
    # Log to wandb
    logger.log({
        "data/n_trajectories": len(records),
        "data/alpha_min": min(alphas),
        "data/alpha_max": max(alphas),
        "data/D0_min": min(D0s),
        "data/D0_max": max(D0s),
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
    
    # Create and upload dataset
    print(f"\nUploading to {repo_id}...")
    dataset = DatasetDict({"train": Dataset.from_list(records)})
    
    dataset.push_to_hub(
        repo_id,
        token=token,
        commit_message=f"Upload FBM dataset ({n_trajectories:,} trajectories)",
    )
    
    logger.set_summary("upload_success", True)
    logger.finish()
    
    print("\n" + "=" * 60)
    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()

