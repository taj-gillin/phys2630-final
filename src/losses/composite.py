import torch


def compute_losses(model, batch_data, batch_phys, batch_ic, batch_bc, lambdas, bc_type: str):
    loss_data = torch.tensor(0.0, device=model.alpha.device)
    loss_phys = torch.tensor(0.0, device=model.alpha.device)
    loss_ic = torch.tensor(0.0, device=model.alpha.device)
    loss_bc = torch.tensor(0.0, device=model.alpha.device)

    if batch_data is not None:
        Xd, ud = batch_data
        pred = model(Xd)
        loss_data = (pred - ud).pow(2).mean()

    if batch_phys is not None:
        Xp = batch_phys
        res = model.physics_residual(Xp)
        loss_phys = res.pow(2).mean()

    if batch_ic is not None:
        Xi, ui = batch_ic
        pred_ic = model(Xi)
        loss_ic = (pred_ic - ui).pow(2).mean()

    if batch_bc is not None:
        Xb, ub = batch_bc
        pred_bc = model(Xb)
        if bc_type == "dirichlet_zero":
            loss_bc = pred_bc.pow(2).mean()
        else:
            loss_bc = (pred_bc - ub).pow(2).mean()

    loss = (
        lambdas["data"] * loss_data
        + lambdas["phys"] * loss_phys
        + lambdas["ic"] * loss_ic
        + lambdas["bc"] * loss_bc
    )

    return {
        "total": loss,
        "data": loss_data.detach(),
        "phys": loss_phys.detach(),
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
    }

