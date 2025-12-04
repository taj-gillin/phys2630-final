from pathlib import Path
import torch
from torch.utils.data import DataLoader

from utils.config import load_config, ensure_dir
from data.load_raw import load_experiment, compute_domain_bounds
from data.preprocess import trajectories_to_density, density_to_samples, initial_condition
from data.sampling import sample_collocation
from data.datasets import ArrayDataset, CollocationDataset
from models.pinn import AnomalousDiffusionPINN
from training.loops import train_with_phases


def run_experiment(config_path: str):
    cfg = load_config(config_path)

    # paths
    out_dir = ensure_dir(cfg["logging"]["output_dir"])

    # load data
    dfs = load_experiment(cfg["data"]["track"], cfg["data"]["exp"], cfg["data"]["fovs"])
    domain = compute_domain_bounds(dfs, margin=cfg["preprocess"]["domain_margin"])

    # build density for each FOV and stack samples
    data_pts = []
    data_vals = []
    for df in dfs.values():
        density, xg, yg, tg = trajectories_to_density(
            df,
            grid_size=cfg["preprocess"]["grid_size"],
            time_bins=cfg["preprocess"]["time_bins"],
            domain_bounds=domain,
            smoothing_sigma=cfg["preprocess"]["smoothing_sigma"],
        )
        X, u = density_to_samples(
            density,
            xg,
            yg,
            tg,
            n_samples=cfg["preprocess"]["sample_data_points"],
        )
        data_pts.append(X)
        data_vals.append(u)

    X_data = torch.tensor(torch.cat([torch.as_tensor(x) for x in data_pts], dim=0), dtype=torch.float32)
    u_data = torch.tensor(torch.cat([torch.as_tensor(u) for u in data_vals], dim=0), dtype=torch.float32)

    collocation = sample_collocation(domain, cfg["preprocess"]["sample_collocation"])
    X_phys = torch.as_tensor(collocation, dtype=torch.float32)

    data_loader = DataLoader(ArrayDataset(X_data, u_data), batch_size=cfg["training"]["batch_data"], shuffle=True)
    phys_loader = DataLoader(CollocationDataset(X_phys), batch_size=cfg["training"]["batch_phys"], shuffle=True)

    model = AnomalousDiffusionPINN(
        hidden_layers=cfg["model"]["hidden_layers"],
        hidden_dim=cfg["model"]["hidden_dim"],
        activation=cfg["model"]["activation"],
        alpha_init=cfg["model"]["alpha_init"],
        D0_init=cfg["model"]["D0_init"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = torch.optim.Adam(model.parameters(), lr=cfg["training"]["phases"][0]["lr"])

    phases = cfg["training"]["phases"]
    history = train_with_phases(
        model,
        optim,
        device,
        phases=phases,
        loaders={"data": data_loader, "phys": phys_loader},
        lambda_ic=cfg["training"]["lambda_ic"],
        bc_type=cfg["training"]["bc_type"],
        grad_clip=cfg["training"]["grad_clip"],
    )

    alpha, D0 = model.learned_parameters()
    with (Path(out_dir) / "metrics.txt").open("w") as f:
        f.write(f"alpha={alpha}\nD0={D0}\n")
        f.write(str(history))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(args.config)

