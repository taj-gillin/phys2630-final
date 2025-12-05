#!/usr/bin/env python3
"""
Train generalized models on a large synthetic dataset.

Uses the modular architecture:
- Encoders: linear, mlp, lstm, cnn, hybrid
- Losses: supervised, physics, combined (PINN)

Usage:
    python scripts/train_models.py --config experiments/exp_train_models.yaml
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

from data.synthetic import generate_fbm_trajectory
from methods.predictor import DiffusionPredictor
from methods.losses import create_loss
from utils.config import load_config, ensure_dir


class TrajectoryDataset(Dataset):
    """Dataset of synthetic FBM trajectories for training."""
    
    def __init__(
        self,
        n_trajectories: int = 100000,
        trajectory_length: int = 200,
        alpha_range: tuple[float, float] = (0.1, 2.0),
        D0_range: tuple[float, float] = (0.1, 10.0),
        seed: Optional[int] = None,
    ):
        self.n_trajectories = n_trajectories
        self.trajectory_length = trajectory_length
        self.alpha_range = alpha_range
        self.D0_range = D0_range
        
        if seed is not None:
            np.random.seed(seed)
        
        print(f"Generating {n_trajectories} synthetic trajectories...")
        self.trajectories = []
        self.alphas = []
        self.D0s = []
        
        for i in tqdm(range(n_trajectories), desc="Generating"):
            alpha = np.random.uniform(*alpha_range)
            D0 = np.random.uniform(*D0_range)
            
            traj = generate_fbm_trajectory(
                alpha=alpha,
                length=trajectory_length,
                D0=D0,
            )
            
            self.trajectories.append(traj.positions)
            self.alphas.append(alpha)
            self.D0s.append(D0)
        
        self.trajectories = np.array(self.trajectories)
        self.alphas = np.array(self.alphas)
        self.D0s = np.array(self.D0s)
        
        print(f"Dataset ready: {n_trajectories} trajectories, length {trajectory_length}")
    
    def __len__(self):
        return self.n_trajectories
    
    def __getitem__(self, idx):
        traj = torch.tensor(self.trajectories[idx], dtype=torch.float32)
        alpha = torch.tensor(self.alphas[idx], dtype=torch.float32)
        D0 = torch.tensor(self.D0s[idx], dtype=torch.float32)
        
        # Normalize trajectory
        traj = traj - traj[0:1, :]
        std = traj.std() + 1e-8
        traj = traj / std
        
        return traj, alpha, D0, std


def train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    model_name: str,
    output_dir: Path,
    device: str = "cuda",
):
    """Train a single model with the given loss function."""
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["lr"] * 0.01,
    )
    
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_alpha_mae": []}
    
    use_physics = hasattr(loss_fn, 'physics') or loss_fn.__class__.__name__ == 'PhysicsLoss'
    
    print(f"\nTraining {model_name}...")
    print(f"  Encoder: {model.encoder_name}")
    print(f"  Loss: {loss_fn.__class__.__name__}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['lr']}")
    
    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        for batch in pbar:
            traj, alpha_true, D0_true, scale = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            
            alpha_pred, D0_pred = model(traj)
            
            # Compute loss (handles both simple and combined losses)
            if hasattr(loss_fn, 'physics'):  # CombinedLoss
                loss, _ = loss_fn(alpha_pred, D0_pred, alpha_true, D0_true, traj, scale)
            else:
                loss = loss_fn(alpha_pred, D0_pred, alpha_true, D0_true, traj, scale)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses = []
        val_alpha_errors = []
        
        with torch.no_grad():
            for batch in val_loader:
                traj, alpha_true, D0_true, scale = [x.to(device) for x in batch]
                
                alpha_pred, D0_pred = model(traj)
                
                # Just use alpha MSE for validation metric
                loss_alpha = torch.nn.functional.mse_loss(alpha_pred, alpha_true)
                val_losses.append(loss_alpha.item())
                
                alpha_mae = torch.abs(alpha_pred - alpha_true).mean()
                val_alpha_errors.append(alpha_mae.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_alpha_mae = np.mean(val_alpha_errors)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_alpha_mae"].append(val_alpha_mae)
        
        print(f"  Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, α_MAE={val_alpha_mae:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_alpha_mae": val_alpha_mae,
                "config": {
                    "encoder_name": model.encoder_name,
                    "encoder_kwargs": {},
                },
            }
            torch.save(checkpoint, output_dir / f"{model_name}_best.pt")
            print(f"    ✓ Saved best model (val_loss={val_loss:.4f})")
    
    # Save final model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": config["epochs"],
        "history": history,
        "config": {
            "encoder_name": model.encoder_name,
            "encoder_kwargs": {},
        },
    }
    torch.save(checkpoint, output_dir / f"{model_name}_final.pt")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train generalized models")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg["output"]["dir"])
    ensure_dir(output_dir)
    
    print("=" * 60)
    print("Generalized Model Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Start time: {datetime.now()}")
    
    # Create datasets
    data_cfg = cfg["data"]
    train_dataset = TrajectoryDataset(
        n_trajectories=data_cfg["n_train"],
        trajectory_length=data_cfg["trajectory_length"],
        alpha_range=tuple(data_cfg["alpha_range"]),
        D0_range=tuple(data_cfg.get("D0_range", [0.1, 10.0])),
        seed=data_cfg.get("seed"),
    )
    
    val_dataset = TrajectoryDataset(
        n_trajectories=data_cfg["n_val"],
        trajectory_length=data_cfg["trajectory_length"],
        alpha_range=tuple(data_cfg["alpha_range"]),
        D0_range=tuple(data_cfg.get("D0_range", [0.1, 10.0])),
        seed=data_cfg.get("seed", 0) + 999999,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )
    
    all_history = {}
    
    # Train each model configuration
    for model_name, model_cfg in cfg["models"].items():
        if not model_cfg.get("enabled", True):
            print(f"\nSkipping {model_name} (disabled)")
            continue
        
        encoder_name = model_cfg.get("encoder", model_name)
        loss_name = model_cfg.get("loss", "supervised")
        
        # Create model
        encoder_kwargs = {k: v for k, v in model_cfg.items() 
                        if k not in ["enabled", "encoder", "loss", "lambda_physics"]}
        
        model = DiffusionPredictor(
            encoder_name=encoder_name,
            encoder_kwargs=encoder_kwargs,
        )
        
        # Create loss
        loss_kwargs = {}
        if loss_name == "combined":
            loss_kwargs["lambda_physics"] = model_cfg.get("lambda_physics", 0.1)
        
        loss_fn = create_loss(loss_name, **loss_kwargs)
        
        # Train
        history = train_model(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            config={**cfg["training"], **model_cfg},
            model_name=model_name,
            output_dir=output_dir,
            device=device,
        )
        all_history[model_name] = history
    
    # Save training summary
    summary = {
        "config": cfg,
        "device": device,
        "history": all_history,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Models saved to: {output_dir}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
