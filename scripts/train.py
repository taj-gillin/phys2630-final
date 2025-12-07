#!/usr/bin/env python3
"""
Train a model to predict α (anomalous diffusion exponent) and/or classify
the diffusion model on HuggingFace trajectory dataset.

Following the AnDi Challenge (https://www.nature.com/articles/s41467-021-26320-w):
- Task 1: Inference of the anomalous diffusion exponent α
- Task 2: Classification of the diffusion model (CTRW, FBM, LW, ATTM, SBM)

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
from data.trajectory import MODEL_NAMES, NUM_MODELS
from methods.predictor import DiffusionPredictor
from methods.losses import create_loss, MultiTaskLoss
from utils.config import load_config, ensure_dir
from utils.wandb_logger import init_wandb


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for HuggingFace trajectories.
    
    Supports both Task 1 (α inference) and Task 2 (model classification).
    """
    
    def __init__(self, hf_dataset, pad_to_length: int = 1000, tasks: list = None):
        self.trajectories = hf_dataset.trajectories
        self.pad_to_length = pad_to_length
        self.tasks = tasks if tasks is not None else ['alpha']
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        positions = torch.tensor(traj.positions, dtype=torch.float32)
        length = len(traj.positions)
        
        # Pad/truncate
        if length < self.pad_to_length:
            padding = torch.zeros(self.pad_to_length - length, 2)
            positions = torch.cat([positions, padding], dim=0)
        elif length > self.pad_to_length:
            positions = positions[:self.pad_to_length]
        
        # Build output dict
        out = {
            'positions': positions,
            'length': torch.tensor(length, dtype=torch.long),
        }
        
        # Task 1: Alpha
        if 'alpha' in self.tasks:
            out['alpha'] = torch.tensor(traj.alpha_true, dtype=torch.float32)
        
        # Task 2: Model classification
        if 'model' in self.tasks:
            # Default to -1 if model not available (will be handled in training)
            model_idx = traj.model_true if traj.model_true is not None else -1
            out['model'] = torch.tensor(model_idx, dtype=torch.long)
        
        return out


def collate_fn(batch):
    """Collate function that handles dict outputs."""
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([b[key] for b in batch])
    return result


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


def compute_metrics(
    alpha_pred: Optional[torch.Tensor],
    alpha_true: Optional[torch.Tensor],
    model_pred: Optional[torch.Tensor],
    model_true: Optional[torch.Tensor],
) -> dict:
    """Compute evaluation metrics for both tasks."""
    metrics = {}
    
    # Task 1 metrics
    if alpha_pred is not None and alpha_true is not None:
        metrics['alpha_mae'] = torch.abs(alpha_pred - alpha_true).mean().item()
        metrics['alpha_mse'] = torch.nn.functional.mse_loss(alpha_pred, alpha_true).item()
    
    # Task 2 metrics
    if model_pred is not None and model_true is not None:
        # Filter out invalid labels (-1)
        valid_mask = model_true >= 0
        if valid_mask.sum() > 0:
            pred_classes = torch.argmax(model_pred[valid_mask], dim=-1)
            true_classes = model_true[valid_mask]
            metrics['model_accuracy'] = (pred_classes == true_classes).float().mean().item()
            
            # Compute per-class accuracy for F1 approximation
            for i, name in enumerate(MODEL_NAMES):
                mask = true_classes == i
                if mask.sum() > 0:
                    metrics[f'model_acc_{name}'] = (pred_classes[mask] == i).float().mean().item()
    
    return metrics


def train(
    model: nn.Module,
    loss_fn: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    output_dir: Path,
    logger,
    device: str,
    tasks: list,
):
    """Train the model (supports single or multi-task)."""
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
    
    # Learning rate scheduler with optional warmup
    scheduler_cfg = train_cfg.get("scheduler", {})
    warmup_epochs = scheduler_cfg.get("warmup_epochs", 5)
    min_lr_factor = scheduler_cfg.get("min_lr_factor", 0.01)
    
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"] - warmup_epochs,
            eta_min=train_cfg["lr"] * min_lr_factor,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"],
            eta_min=train_cfg["lr"] * min_lr_factor,
        )
    
    # Track best model
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_cfg = train_cfg.get("early_stopping", {})
    patience = early_stop_cfg.get("patience", 10)
    min_delta = early_stop_cfg.get("min_delta", 0.0001)
    
    # Extended history for multi-task
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_alpha_mae": [],
        "val_model_acc": [],
    }
    
    # Log model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_model_summary(
        encoder_name=model.encoder_name,
        num_params=num_params,
        loss_type=loss_fn.__class__.__name__,
    )
    
    print(f"\nModel: {model.encoder_name.upper()}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Parameters: {num_params:,}")
    print(f"Loss: {loss_fn.__class__.__name__}")
    print(f"Epochs: {train_cfg['epochs']}")
    print(f"Learning rate: {train_cfg['lr']}")
    if warmup_epochs > 0:
        print(f"Scheduler: {warmup_epochs} epoch warmup → cosine decay")
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
    is_multitask = len(tasks) > 1
    
    for epoch in range(train_cfg["epochs"]):
        # Training
        model.train()
        train_losses = []
        train_loss_breakdowns = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            traj = batch['positions'].to(device)
            lengths = batch['length'].to(device)
            alpha_true = batch.get('alpha')
            model_true = batch.get('model')
            
            if alpha_true is not None:
                alpha_true = alpha_true.to(device)
            if model_true is not None:
                model_true = model_true.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            output = model(traj)
            
            # Handle single vs multi-task outputs
            if is_multitask:
                alpha_pred, model_pred = output
            else:
                if 'alpha' in tasks:
                    alpha_pred = output
                    model_pred = None
                else:
                    alpha_pred = None
                    model_pred = output

            # Compute loss
            if isinstance(loss_fn, MultiTaskLoss):
                loss, loss_breakdown = loss_fn(
                    alpha_pred, alpha_true,
                    model_pred, model_true,
                    traj, lengths
                )
            else:
                # Single-task loss
                if 'alpha' in tasks:
                    loss_output = loss_fn(alpha_pred, alpha_true, traj, lengths)
                else:
                    loss_output = loss_fn(model_pred, model_true, traj, lengths)
                
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
        val_model_correct = []
        val_model_total = []
        
        with torch.no_grad():
            for batch in val_loader:
                traj = batch['positions'].to(device)
                lengths = batch['length'].to(device)
                alpha_true = batch.get('alpha')
                model_true = batch.get('model')
                
                if alpha_true is not None:
                    alpha_true = alpha_true.to(device)
                if model_true is not None:
                    model_true = model_true.to(device)
                
                output = model(traj)
                
                if is_multitask:
                    alpha_pred, model_pred = output
                else:
                    if 'alpha' in tasks:
                        alpha_pred = output
                        model_pred = None
                    else:
                        alpha_pred = None
                        model_pred = output
                
                # Compute validation loss (for early stopping)
                if alpha_pred is not None and alpha_true is not None:
                    val_loss = torch.nn.functional.mse_loss(alpha_pred, alpha_true)
                    val_losses.append(val_loss.item())
                    alpha_mae = torch.abs(alpha_pred - alpha_true).mean()
                    val_alpha_errors.append(alpha_mae.item())
                
                if model_pred is not None and model_true is not None:
                    valid_mask = model_true >= 0
                    if valid_mask.sum() > 0:
                        pred_classes = torch.argmax(model_pred[valid_mask], dim=-1)
                        true_classes = model_true[valid_mask]
                        correct = (pred_classes == true_classes).sum().item()
                        total = valid_mask.sum().item()
                        val_model_correct.append(correct)
                        val_model_total.append(total)
                        
                        # Use classification loss for early stopping if no alpha
                        if alpha_pred is None:
                            ce_loss = torch.nn.functional.cross_entropy(
                                model_pred[valid_mask], true_classes
                            )
                            val_losses.append(ce_loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else 0.0
        val_alpha_mae = np.mean(val_alpha_errors) if val_alpha_errors else 0.0
        val_model_acc = sum(val_model_correct) / max(sum(val_model_total), 1) if val_model_total else 0.0
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
        history["val_model_acc"].append(val_model_acc)

        # Log to wandb
        log_metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "learning_rate": current_lr,
        }
        if 'alpha' in tasks:
            log_metrics["val/alpha_mae"] = val_alpha_mae
        if 'model' in tasks:
            log_metrics["val/model_accuracy"] = val_model_acc
        
        if train_loss_breakdown_avg:
            for key, value in train_loss_breakdown_avg.items():
                log_metrics[f"train/{key}"] = value
        
        logger.log(log_metrics)
        
        # Print progress
        status = f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}"
        if 'alpha' in tasks:
            status += f", α_MAE={val_alpha_mae:.4f}"
        if 'model' in tasks:
            status += f", model_acc={val_model_acc:.3f}"
        print(status)
        
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
                "val_model_acc": val_model_acc,
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
    if 'alpha' in tasks:
        logger.set_summary("final_val_alpha_mae", val_alpha_mae)
    if 'model' in tasks:
        logger.set_summary("final_val_model_acc", val_model_acc)
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
    
    # Determine tasks from config
    model_cfg = config["model"]
    tasks = model_cfg.get("tasks", ["alpha"])
    if isinstance(tasks, str):
        tasks = [tasks]
    
    # Header
    print("=" * 60)
    print(f"Training: {exp_name}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Output: {output_dir}")
    print(f"Start: {datetime.now()}")
    if logger.run_url:
        print(f"Wandb: {logger.run_url}")
    
    # Load data from HuggingFace
    data_cfg = config["data"]
    train_split = data_cfg.get("train_split", data_cfg.get("split", "train"))
    val_split = data_cfg.get("val_split", "validation")
    
    print("\nLoading training data from HuggingFace...")
    train_data = load_from_hf(
        repo_id=data_cfg["repo_id"],
        split=train_split,
        alpha_range=tuple(data_cfg.get("alpha_range", [0.1, 2.0])),
        length_range=tuple(data_cfg.get("length_range", [50, 1000])),
        max_trajectories=data_cfg.get("n_train"),
        shuffle=True,
        seed=seed,
    )
    print(f"  Loaded {len(train_data)} training trajectories")
    
    # Log model distribution if available
    if 'model' in tasks:
        model_counts = train_data.model_counts
        if model_counts:
            print(f"  Model distribution: {model_counts}")
    
    print("Loading validation data...")
    val_data = load_from_hf(
        repo_id=data_cfg["repo_id"],
        split=val_split,
        alpha_range=tuple(data_cfg.get("alpha_range", [0.1, 2.0])),
        length_range=tuple(data_cfg.get("length_range", [50, 1000])),
        max_trajectories=data_cfg.get("n_val"),
        shuffle=False,
        seed=seed,
    )
    print(f"  Loaded {len(val_data)} validation trajectories")
    
    # Create data loaders
    pad_length = data_cfg.get("pad_to_length", 1000)
    train_dataset = TrajectoryDataset(train_data, pad_to_length=pad_length, tasks=tasks)
    val_dataset = TrajectoryDataset(val_data, pad_to_length=pad_length, tasks=tasks)
    
    train_cfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Create model
    encoder_name = model_cfg["encoder"]
    encoder_params = model_cfg.get("params", {})
    
    # For linear/mlp, set seq_len
    if encoder_name in ["linear", "mlp"]:
        encoder_params["seq_len"] = pad_length
    
    model = DiffusionPredictor(
        encoder_name=encoder_name,
        encoder_kwargs=encoder_params,
        num_models=NUM_MODELS,
        tasks=tasks,
    )
    
    # Create loss
    loss_name = model_cfg.get("loss", "supervised")
    loss_params = model_cfg.get("loss_params", {})
    
    # For multi-task, use MultiTaskLoss by default
    if len(tasks) > 1 and loss_name not in ["multitask"]:
        loss_name = "multitask"
        print(f"  Using MultiTaskLoss for multi-task training")
    
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
        tasks=tasks,
    )
    
    # Save config and summary
    with open(output_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    summary = {
        "experiment": exp_name,
        "tasks": tasks,
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
