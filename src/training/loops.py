from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from tqdm import trange


def train_with_phases(model: torch.nn.Module, optim: torch.optim.Optimizer, device: torch.device, phases: List[Dict], loaders: Dict[str, DataLoader], lambda_ic: float, bc_type: str, grad_clip: float):
    history = []
    model.to(device)
    for phase in phases:
        epochs = phase["epochs"]
        lambda_data = phase["lambda_data"]
        lambda_phys = phase["lambda_phys"]
        pbar = trange(epochs, desc=phase["name"], leave=False)
        for _ in pbar:
            model.train()
            total_loss = 0.0

            # data loss
            if "data" in loaders:
                for X, u in loaders["data"]:
                    X = X.to(device)
                    u = u.to(device)
                    optim.zero_grad()
                    pred = model(X)
                    loss_data = (pred - u).pow(2).mean()
                    loss_phys = 0.0
                    if "phys" in loaders and lambda_phys > 0:
                        Xp = next(iter(loaders["phys"])).to(device)
                        res = model.physics_residual(Xp)
                        loss_phys = res.pow(2).mean()
                    loss = lambda_data * loss_data + lambda_phys * loss_phys
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optim.step()
                    total_loss += loss.item()

            pbar.set_postfix(loss=total_loss)
        history.append({"phase": phase["name"], "loss": total_loss})
    return history

