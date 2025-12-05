#!/usr/bin/env python3
"""
Generate FBM trajectories and upload to Hugging Face Hub.

Usage:
    python scripts/upload_dataset.py --token <HF_TOKEN>
    
Or set HF_TOKEN environment variable:
    export HF_TOKEN=<your_token>
    python scripts/upload_dataset.py
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from andi_datasets.models_theory import models_theory

# Dataset configuration
REPO_ID = "taj-gillin/andi-trajectory"
ALPHAS = [0.3, 0.5, 0.7, 0.9, 1.0]
TRAJECTORIES_PER_ALPHA = 50
LENGTHS = [100, 200, 500]
D0 = 1.0
SEED = 42


def generate_trajectory(alpha: float, length: int, D0: float, seed: int) -> tuple[list, list]:
    """Generate a single 2D FBM trajectory."""
    np.random.seed(seed)
    MT = models_theory()
    
    traj_x = MT.fbm(T=length, alpha=alpha)
    traj_y = MT.fbm(T=length, alpha=alpha)
    
    # Scale by diffusion coefficient
    scale = np.sqrt(D0)
    x = (traj_x * scale).tolist()
    y = (traj_y * scale).tolist()
    
    return x, y


def generate_all_trajectories():
    """Generate all trajectories according to specification."""
    print("Generating trajectories...")
    print(f"  Alphas: {ALPHAS}")
    print(f"  Lengths: {LENGTHS}")
    print(f"  Trajectories per (alpha, length): {TRAJECTORIES_PER_ALPHA}")
    
    records = []
    traj_id = 0
    
    total = len(ALPHAS) * len(LENGTHS) * TRAJECTORIES_PER_ALPHA
    
    for alpha in ALPHAS:
        for length in LENGTHS:
            for i in range(TRAJECTORIES_PER_ALPHA):
                # Unique seed for each trajectory
                seed = SEED + traj_id
                
                x, y = generate_trajectory(alpha, length, D0, seed)
                
                records.append({
                    "trajectory_id": traj_id,
                    "alpha": alpha,
                    "D0": D0,
                    "length": length,
                    "x": x,
                    "y": y,
                })
                
                traj_id += 1
                
                if traj_id % 100 == 0:
                    print(f"  Generated {traj_id}/{total} trajectories...")
    
    print(f"  Total: {len(records)} trajectories")
    return records


def create_dataset(records: list) -> DatasetDict:
    """Create HuggingFace Dataset from records."""
    # Create a single train split (we'll filter by alpha/length in experiments)
    dataset = Dataset.from_list(records)
    
    # Create DatasetDict with train split
    dataset_dict = DatasetDict({"train": dataset})
    
    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Upload FBM trajectories to Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--dry-run", action="store_true", help="Generate but don't upload")
    args = parser.parse_args()
    
    # Get token from args, environment, or HF CLI login
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token and not args.dry_run:
        # Try to get token from HF CLI login
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    
    if not token and not args.dry_run:
        print("Error: HF token required.")
        print("Options:")
        print("  1. Login with: huggingface-cli login")
        print("  2. Use --token <your_token>")
        print("  3. Set HF_TOKEN environment variable")
        sys.exit(1)
    
    print("=" * 60)
    print("FBM Trajectory Dataset Generator")
    print("=" * 60)
    print(f"Repository: {REPO_ID}")
    print()
    
    # Generate trajectories
    records = generate_all_trajectories()
    
    # Create dataset
    print("\nCreating HuggingFace Dataset...")
    dataset = create_dataset(records)
    print(f"  Dataset: {dataset}")
    
    if args.dry_run:
        print("\n[DRY RUN] Skipping upload.")
        print("Sample record:")
        sample = records[0]
        print(f"  trajectory_id: {sample['trajectory_id']}")
        print(f"  alpha: {sample['alpha']}")
        print(f"  length: {sample['length']}")
        print(f"  x[:5]: {sample['x'][:5]}")
        return
    
    # Upload to Hub
    print(f"\nUploading to {REPO_ID}...")
    dataset.push_to_hub(
        REPO_ID,
        token=token,
        commit_message="Upload FBM trajectory dataset (750 trajectories)",
    )
    
    print("\n" + "=" * 60)
    print("Upload complete!")
    print(f"View at: https://huggingface.co/datasets/{REPO_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()

