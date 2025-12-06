#!/usr/bin/env python3
"""
Train a model to predict α (anomalous diffusion exponent) on HuggingFace trajectory dataset.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w),
we focus on Task 1: Inference of the anomalous diffusion exponent α.

Usage:
    python scripts/train.py --config configs/models/lstm.yaml
    sbatch slurm/submit.slurm train configs/models/lstm.yaml
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.hf_loader import load_from_hf
from methods.predictor import DiffusionPredictor
from methods.losses import create_loss
from utils.config import load_config, ensure_dir
from utils.wandb_logger import init_wandb


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for HuggingFace trajectories."""
    
    def __init__(self, hf_dataset, pad_to_length: int = 1000):
        self.trajectories = hf_dataset.trajectories
        self.pad_to_length = pad_to_length
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        positions = torch.tensor(traj.positions, dtype=torch.float32)
        alpha = torch.tensor(traj.alpha_true, dtype=torch.float32)
        length = len(traj.positions)
        
        # Pad/truncate
        if length < self.pad_to_length:
            padding = torch.zeros(self.pad_to_length - length, 2)
            positions = torch.cat([positions, padding], dim=0)
        elif length > self.pad_to_length:
            positions = positions[:self.pad_to_length]
        
        return positions, alpha, torch.tensor(length, dtype=torch.long)


def get_device(config: dict) -> str:
    """Get device from config."""
    device_cfg = config.get("env", {}).get("device", "auto")
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def compute_regularization(
    model: nn.Module, l1_coeff: float, l2_coeff: float
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Compute optional L1/L2 regularization loss for the model."""
    if not (l1_coeff or l2_coeff):
        return None, {}

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None, {}

    device = params[0].device
    reg_loss = torch.tensor(0.0, device=device)
    breakdown: dict[str, float] = {}

    if l1_coeff:
        l1_norm = sum(p.abs().sum() for p in params)
        l1_term = l1_coeff * l1_norm
        reg_loss = reg_loss + l1_term
        breakdown["l1"] = l1_term.item()

    if l2_coeff:
        l2_norm = sum(p.pow(2).sum() for p in params)
        l2_term = l2_coeff * l2_norm
        reg_loss = reg_loss + l2_term
        breakdown["l2"] = l2_term.item()

    return reg_loss, breakdown


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    output_dir: Path,
    logger,
    device: str,
):
    """Train the model."""
    model = model.to(device)
    
    train_cfg = config["training"]
    exp_name = config["experiment"]["name"]

    regularization_cfg = train_cfg.get("regularization", {})
    l1_coeff = float(regularization_cfg.get("l1", 0.0))
    l2_coeff = float(regularization_cfg.get("l2", 0.0))
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=train_cfg["lr"] * 0.01,
    )
    
    # Track best model
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_cfg = train_cfg.get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 10)
    min_delta = early_stop_cfg.get("min_delta", 0.0001)
    
    history = {"train_loss": [], "val_loss": [], "val_alpha_mae": []}
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_model_summary(
        encoder_name=model.encoder_name,
        num_params=num_params,
        loss_type=loss_fn.__class__.__name__,
    )
    
    print(f"\nModel: {model.encoder_name.upper()}")
    print(f"Parameters: {num_params:,}")
    print(f"Loss: {loss_fn.__class__.__name__}")
    print(f"Epochs: {train_cfg['epochs']}")
    print(f"Learning rate: {train_cfg['lr']}")
    if l1_coeff or l2_coeff:
        reg_parts = []
        if l1_coeff:
            reg_parts.append(f"L1={l1_coeff}")
        if l2_coeff:
            reg_parts.append(f"L2={l2_coeff}")
        print(f"Regularization: {', '.join(reg_parts)}")
    print()
    
    log_interval = train_cfg.get("log_interval", 10)
    total_batches = len(train_loader)
    
    for epoch in range(train_cfg["epochs"]):
        # Training
        model.train()
        train_losses = []
        train_loss_breakdowns = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            traj, alpha_true, lengths = [x.to(device) for x in batch]

            optimizer.zero_grad()
            alpha_pred = model(traj)

            # Compute loss - some loss functions return breakdown dict
            loss_output = loss_fn(alpha_pred, alpha_true, traj, lengths)
            if isinstance(loss_output, tuple):
                loss, loss_breakdown = loss_output
            else:
                loss = loss_output
                loss_breakdown = None

            batch_reg_breakdown: dict[str, float] = {}
            if l1_coeff or l2_coeff:
                reg_loss, batch_reg_breakdown = compute_regularization(
                    model, l1_coeff, l2_coeff
                )
                if reg_loss is not None:
                    loss = loss + reg_loss
                    if loss_breakdown is None:
                        loss_breakdown = {}
                    loss_breakdown.update(batch_reg_breakdown)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            if loss_breakdown:
                train_loss_breakdowns.append(loss_breakdown)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if batch_idx % log_interval == 0:
                metrics = {
                    "train/batch_loss": loss.item(),
                    "train/step": epoch * total_batches + batch_idx,
                }
                for reg_name, reg_value in batch_reg_breakdown.items():
                    metrics[f"train/regularization/{reg_name}"] = reg_value
                logger.log(metrics)
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        val_alpha_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                traj, alpha_true, lengths = [x.to(device) for x in batch]
                alpha_pred = model(traj)
                
                val_loss = torch.nn.functional.mse_loss(alpha_pred, alpha_true)
                val_losses.append(val_loss.item())
                
                alpha_mae = torch.abs(alpha_pred - alpha_true).mean()
                val_alpha_errors.append(alpha_mae.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_alpha_mae = np.mean(val_alpha_errors)
        current_lr = scheduler.get_last_lr()[0]

        # Compute average loss breakdown if available
        train_loss_breakdown_avg = None
        if train_loss_breakdowns:
            train_loss_breakdown_avg = {}
            for key in train_loss_breakdowns[0].keys():
                train_loss_breakdown_avg[key] = np.mean([bd[key] for bd in train_loss_breakdowns])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_alpha_mae"].append(val_alpha_mae)

        # Log to wandb
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_alpha_mae=val_alpha_mae,
            learning_rate=current_lr,
            train_loss_breakdown=train_loss_breakdown_avg,
        )
        
        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, α_MAE={val_alpha_mae:.4f}")
        
        # Save best model
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_alpha_mae": val_alpha_mae,
                "config": config,
            }
            best_path = output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
            
            logger.save_artifact(
                path=best_path,
                name=f"{exp_name}_best",
                metadata={"val_loss": val_loss, "epoch": epoch},
                aliases=["best"],
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stop_cfg.get("enabled", False) and patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save final model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "history": history,
        "config": config,
    }
    torch.save(checkpoint, output_dir / "final.pt")
    
    # Log final summary
    logger.set_summary("best_val_loss", best_val_loss)
    logger.set_summary("final_val_alpha_mae", val_alpha_mae)
    logger.set_summary("total_epochs", epoch + 1)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train a diffusion predictor model")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    exp_name = config["experiment"]["name"]
    device = get_device(config)
    
    # Set seed
    seed = config.get("env", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Output directory based on experiment name
    output_dir = Path("outputs") / exp_name
    ensure_dir(output_dir)
    
    # Initialize wandb
    logger = init_wandb(config, job_type="train")
    
    # Header
    print("=" * 60)
    print(f"Training: {exp_name}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Start: {datetime.now()}")
    if logger.run_url:
        print(f"Wandb: {logger.run_url}")
    
    # Load data from HuggingFace
    data_cfg = config["data"]
    
    print("\nLoading training data from HuggingFace...")
    train_data = load_from_hf(
        repo_id=data_cfg["repo_id"],
        alpha_range=tuple(data_cfg.get("alpha_range", [0.1, 2.0])),
        length_range=tuple(data_cfg.get("length_range", [50, 1000])),
        max_trajectories=data_cfg.get("n_train"),
        shuffle=True,
        seed=seed,
    )
    print(f"  Loaded {len(train_data)} training trajectories")
    
    print("Loading validation data...")
    val_data = load_from_hf(
        repo_id=data_cfg["repo_id"],
        alpha_range=tuple(data_cfg.get("alpha_range", [0.1, 2.0])),
        length_range=tuple(data_cfg.get("length_range", [50, 1000])),
        max_trajectories=data_cfg.get("n_val"),
        shuffle=True,
        seed=seed + 999999,
    )
    print(f"  Loaded {len(val_data)} validation trajectories")
    
    # Create data loaders
    pad_length = data_cfg.get("pad_to_length", 1000)
    train_dataset = TrajectoryDataset(train_data, pad_to_length=pad_length)
    val_dataset = TrajectoryDataset(val_data, pad_to_length=pad_length)
    
    train_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    
    # Create model
    model_cfg = config["model"]
    encoder_name = model_cfg["encoder"]
    encoder_params = model_cfg.get("params", {})
    
    # For linear/mlp, set seq_len
    if encoder_name in ["linear", "mlp"]:
        encoder_params["seq_len"] = pad_length
    
    model = DiffusionPredictor(
        encoder_name=encoder_name,
        encoder_kwargs=encoder_params,
    )
    
    # Create loss
    loss_name = model_cfg.get("loss", "supervised")
    loss_params = model_cfg.get("loss_params", {})
    loss_fn = create_loss(loss_name, **loss_params)
    
    # Train
    history = train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        logger=logger,
        device=device,
    )
    
    # Save config and summary
    with open(output_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    summary = {
        "experiment": exp_name,
        "history": history,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "wandb_url": logger.run_url,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.finish()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"End: {datetime.now()}")


if __name__ == "__main__":
    main()
