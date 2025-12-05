from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from losses.composite import compute_losses
import itertools


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def train_with_phases(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    device: torch.device,
    phases: List[Dict],
    loaders: Dict[str, DataLoader],
    lambda_ic: float,
    bc_type: str,
    grad_clip: float,
):
    history = []
    model.to(device)
    for phase in phases:
        epochs = phase["epochs"]
        lambdas = {
            "data": phase["lambda_data"],
            "phys": phase["lambda_phys"],
            "ic": lambda_ic,
            "bc": 1.0,
        }
        data_loader = loaders.get("data")
        phys_loader = loaders.get("phys")
        ic_loader = loaders.get("ic")
        bc_loader = loaders.get("bc")

        data_iter = _cycle(data_loader) if data_loader else itertools.repeat(None)
        phys_iter = _cycle(phys_loader) if phys_loader else itertools.repeat(None)
        ic_iter = _cycle(ic_loader) if ic_loader else itertools.repeat(None)
        bc_iter = _cycle(bc_loader) if bc_loader else itertools.repeat(None)

        steps = max(len(data_loader) if data_loader else 0, len(phys_loader) if phys_loader else 0, 1)

        pbar = trange(epochs, desc=phase["name"], leave=False)
        for _ in pbar:
            model.train()
            epoch_loss = 0.0
            for _ in range(steps):
                batch_data = next(data_iter)
                batch_phys = next(phys_iter)
                batch_ic = next(ic_iter)
                batch_bc = next(bc_iter)

                # move to device
                if batch_data is not None:
                    batch_data = (batch_data[0].to(device), batch_data[1].to(device))
                if batch_phys is not None:
                    batch_phys = batch_phys.to(device)
                if batch_ic is not None:
                    batch_ic = (batch_ic[0].to(device), batch_ic[1].to(device))
                if batch_bc is not None:
                    batch_bc = (batch_bc[0].to(device), batch_bc[1].to(device))

                optim.zero_grad()
                losses = compute_losses(model, batch_data, batch_phys, batch_ic, batch_bc, lambdas, bc_type)
                losses["total"].backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
                epoch_loss += float(losses["total"].item())
            pbar.set_postfix(loss=epoch_loss / steps if steps else 0.0)
        history.append({"phase": phase["name"], "loss": epoch_loss / steps if steps else 0.0})
    return history

