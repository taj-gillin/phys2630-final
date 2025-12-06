"""
Weights & Biases integration for experiment tracking.

All settings are read from the experiment config YAML.
"""

import os
from pathlib import Path
from typing import Any, Optional, Union
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbLogger:
    """
    Unified wandb logger that reads all settings from config.
    
    Handles initialization, metric logging, artifact management, and graceful
    degradation when wandb is not available or disabled.
    """
    
    def __init__(
        self,
        config: dict,
        job_type: str = "train",
    ):
        """
        Initialize wandb from experiment config.
        
        Args:
            config: Full experiment configuration dict
            job_type: Type of job (train, compare, generate)
        """
        self.config = config
        self.run = None
        
        # Get wandb config section
        wandb_cfg = config.get("wandb", {})
        exp_cfg = config.get("experiment", {})
        
        # Check if enabled
        self.enabled = (
            WANDB_AVAILABLE and 
            wandb_cfg.get("enabled", True) and
            os.environ.get("WANDB_MODE") != "disabled"
        )
        
        if not self.enabled:
            if not WANDB_AVAILABLE:
                print("[wandb] Not installed - logging disabled")
            elif not wandb_cfg.get("enabled", True):
                print("[wandb] Disabled in config")
            else:
                print("[wandb] Disabled via WANDB_MODE=disabled")
            return
        
        # Build run name from experiment name
        exp_name = exp_cfg.get("name", "experiment")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        
        run_name = f"{exp_name}_{timestamp}"
        if slurm_job_id:
            run_name = f"{exp_name}_{slurm_job_id}"
        
        # Build tags
        tags = list(wandb_cfg.get("tags", []))
        if slurm_job_id:
            tags.append(f"slurm:{slurm_job_id}")
        tags.append(job_type)
        
        # Get mode from environment or default to online
        mode = os.environ.get("WANDB_MODE", "online")
        
        try:
            self.run = wandb.init(
                project=wandb_cfg.get("project", "anomalous-diffusion"),
                entity=wandb_cfg.get("entity"),
                name=run_name,
                config=config,
                tags=tags,
                notes=wandb_cfg.get("notes"),
                group=wandb_cfg.get("group"),
                job_type=job_type,
                mode=mode,
                reinit=True,
            )
            print(f"[wandb] Run: {self.run.url}")
        except Exception as e:
            print(f"[wandb] Failed to initialize: {e}")
            self.enabled = False
    
    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if not self.enabled or self.run is None:
            return
        wandb.log(metrics, step=step)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_alpha_mae: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Log epoch-level training metrics."""
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/alpha_mae": val_alpha_mae,
        }
        if learning_rate is not None:
            metrics["train/lr"] = learning_rate
        self.log(metrics)
    
    def log_model_summary(
        self,
        encoder_name: str,
        num_params: int,
        loss_type: str,
    ) -> None:
        """Log model architecture summary."""
        if not self.enabled:
            return
        self.log({
            "model/encoder": encoder_name,
            "model/params": num_params,
            "model/loss": loss_type,
        })
    
    def save_artifact(
        self,
        path: Union[str, Path],
        name: str,
        artifact_type: str = "model",
        metadata: Optional[dict] = None,
        aliases: Optional[list[str]] = None,
    ) -> None:
        """Save file as wandb artifact."""
        if not self.enabled or self.run is None:
            return
        try:
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=metadata or {},
            )
            artifact.add_file(str(path))
            self.run.log_artifact(artifact, aliases=aliases)
        except Exception as e:
            print(f"[wandb] Failed to save artifact: {e}")
    
    def set_summary(self, key: str, value: Any) -> None:
        """Set a summary metric."""
        if not self.enabled or self.run is None:
            return
        wandb.run.summary[key] = value
    
    def finish(self) -> None:
        """Finish the wandb run."""
        if not self.enabled or self.run is None:
            return
        try:
            wandb.finish()
            print("[wandb] Run finished")
        except Exception as e:
            print(f"[wandb] Error finishing: {e}")
    
    @property
    def run_url(self) -> Optional[str]:
        """Get the URL of the current run."""
        return self.run.url if self.run else None
    
    @property
    def run_id(self) -> Optional[str]:
        """Get the ID of the current run."""
        return self.run.id if self.run else None


def init_wandb(config: dict, job_type: str = "train") -> WandbLogger:
    """Initialize wandb logger from config."""
    return WandbLogger(config=config, job_type=job_type)
