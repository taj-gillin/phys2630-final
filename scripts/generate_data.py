#!/usr/bin/env python3
"""
Generate anomalous diffusion trajectories and upload to HuggingFace Hub.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w):
- Supports all 5 theoretical diffusion models (CTRW, FBM, LW, ATTM, SBM)
- Supports configurable localization noise (SNR levels)
- Task 1: α inference
- Task 2: Model classification

Usage:
    python scripts/generate_data.py --config configs/data/generate.yaml
    sbatch slurm/submit.slurm generate
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.trajectory import MODEL_NAMES, MODEL_TO_IDX, NUM_MODELS
from data.synthetic import generate_trajectory, add_localization_noise, get_valid_alpha_range
from utils.config import load_config
from utils.wandb_logger import init_wandb


def generate_single_trajectory(
    traj_id: int,
    seed: int,
    model: str,
    alpha_range: tuple,
    length_range: tuple,
    snr: Optional[float] = None,
    dimension: int = 2,
) -> dict:
    """
    Generate a single trajectory following AnDi Challenge conventions.
    
    Returns a dictionary suitable for HuggingFace dataset.
    """
    rng = np.random.RandomState(seed + traj_id)
    
    # Sample alpha from range (will be clamped per model)
    alpha = rng.uniform(*alpha_range)
    length = rng.randint(length_range[0], length_range[1] + 1)
    
    # Generate trajectory
    traj = generate_trajectory(
        model=model,
        alpha=alpha,
        length=length,
        dimension=dimension,
        snr=snr,
        seed=seed + traj_id,
    )
    
    return {
        "trajectory_id": traj_id,
        "alpha": float(traj.alpha_true),
        "model": MODEL_TO_IDX[model.upper()],
        "model_name": model.upper(),
        "length": int(length),
        "snr": float(snr) if snr is not None else None,
        "x": traj.positions[:, 0].tolist(),
        "y": traj.positions[:, 1].tolist(),
    }


def generate_batch(
    start_idx: int,
    batch_size: int,
    seed: int,
    models: List[str],
    alpha_range: tuple,
    length_range: tuple,
    snr_levels: Optional[List[float]],
    dimension: int,
    balanced_models: bool,
) -> list:
    """Generate a batch of trajectories."""
    results = []
    rng = np.random.RandomState(seed + start_idx)
    
    for i in range(batch_size):
        traj_id = start_idx + i
        
        # Select model
        if balanced_models:
            # Rotate through models
            model = models[i % len(models)]
        else:
            model = rng.choice(models)
        
        # Select SNR
        if snr_levels:
            snr = rng.choice(snr_levels)
        else:
            snr = None
        
        record = generate_single_trajectory(
            traj_id=traj_id,
            seed=seed,
            model=model,
            alpha_range=alpha_range,
            length_range=length_range,
            snr=snr,
            dimension=dimension,
        )
        results.append(record)
    
    return results


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
    
    # Model configuration
    models = gen_cfg.get("models", MODEL_NAMES)
    if isinstance(models, str):
        models = [models]
    models = [m.upper() if isinstance(m, str) else MODEL_NAMES[m] for m in models]
    balanced_models = gen_cfg.get("balanced_models", True)
    
    # Noise configuration
    snr_levels = gen_cfg.get("snr_levels")
    if snr_levels and not isinstance(snr_levels, list):
        snr_levels = [snr_levels]
    
    dimension = gen_cfg.get("dimension", 2)
    
    n_workers = proc_cfg.get("n_workers", 8)
    batch_size = proc_cfg.get("batch_size", 1000)
    
    # Initialize wandb
    logger = init_wandb(config, job_type="generate")
    
    print("=" * 60)
    print("Anomalous Diffusion Trajectory Dataset Generation")
    print("Following AnDi Challenge Conventions")
    print("=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Trajectories: {n_trajectories:,}")
    print(f"Models: {models}")
    print(f"Alpha range: {alpha_range}")
    print(f"Length range: {length_range}")
    print(f"SNR levels: {snr_levels or 'No noise'}")
    print(f"Dimension: {dimension}D")
    print(f"Balanced models: {balanced_models}")
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
                generate_batch,
                start_idx,
                actual_size,
                seed,
                models,
                alpha_range,
                length_range,
                snr_levels,
                dimension,
                balanced_models,
            )
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            records.extend(future.result())
    
    records.sort(key=lambda x: x["trajectory_id"])
    
    # Statistics
    alphas = [r["alpha"] for r in records]
    lengths = [r["length"] for r in records]
    model_counts = {}
    for r in records:
        model_name = r["model_name"]
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
    
    print(f"\nGenerated {len(records):,} trajectories")
    print(f"Alpha:  [{min(alphas):.3f}, {max(alphas):.3f}], mean={np.mean(alphas):.3f}")
    print(f"Length: [{min(lengths)}, {max(lengths)}], mean={np.mean(lengths):.1f}")
    print(f"Model distribution: {model_counts}")
    
    # Log to wandb
    log_data = {
        "data/n_trajectories": len(records),
        "data/alpha_min": min(alphas),
        "data/alpha_max": max(alphas),
        "data/alpha_mean": np.mean(alphas),
        "data/n_models": len(models),
    }
    for model_name, count in model_counts.items():
        log_data[f"data/model_{model_name}"] = count
    logger.log(log_data)
    
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
            f"Upload multi-model dataset ({', '.join(models)}) with splits "
            f"train={len(train_ds):,}, val={len(val_ds) if val_ds else 0:,}, "
            f"test={len(test_ds) if test_ds else 0:,}"
        ),
    )
    
    logger.set_summary("upload_success", True)
    logger.set_summary("n_models", len(models))
    logger.set_summary("models", models)
    logger.finish()
    
    print("\n" + "=" * 60)
    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
