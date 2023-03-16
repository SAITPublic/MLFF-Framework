import torch
from torch import nn

from ocpmodels.common import distutils
from ocpmodels.modules.loss import L2MAELoss, AtomwiseL2Loss

from src.common.utils import bm_logging # benchmark logging

"""
Available loss

Force:
    mae or l1 : nn.L1Loss 
    mse : nn.MSELoss
    l2mae : ocpmodels.modules.loss.L2MAELoss 
    atomwisel2 : ocpmodels.modules.loss.AtomwiseL2Loss 
                (precisely, AtomwiseL2MAELoss, i.e., L2MAELoss multiplying num of atoms for each snapshot)
    
Energy:
    energy_per_atom_mse : EnergyPerAtomMSELoss

"""

def initiate_loss(loss_name):
    if loss_name in ["l1", "mae"]:
        return DDPLoss(nn.L1Loss())
    elif loss_name == "mse":
        return DDPLoss(nn.MSELoss())
    elif loss_name == "l2mae":
        return DDPLoss(L2MAELoss())
    elif loss_name == "atomwisel2":
        return DDPLoss(AtomwiseL2Loss())
    elif loss_name in ["mae_per_atom", "mse_per_atom", "energy_per_atom_mae", "energy_per_atom_mse"]:
        for name in ["mae", "mse"]:
            if loss_name == f"{name}_per_atom":
                bm_logging.warning(f"`{name}_per_atom` is deprecated later. Please use `energy_per_atom_{name}` instead.")
        if "mae" in loss_name:
            return DDPLoss(EnergyPerAtomMAELoss())
        elif "mse" in loss_name:
            return DDPLoss(EnergyPerAtomMSELoss())
    elif loss_name in ["mae_per_dim", "mse_per_dim", "force_per_dim_mae", "force_per_dim_mse"]:
        for name in ["mae", "mse"]:
            if loss_name == f"{name}_per_dim":
                bm_logging.warning(f"`{name}_per_dim` is deprecated later. Please use `force_per_dim_{name}` instead.")
        if "mae" in loss_name:
            return DDPLoss(nn.L1Loss(), reduction="mean_over_dim")
        elif "mse" in loss_name:
            return DDPLoss(nn.MSELoss(), reduction="mean_over_dim")
    else:
        raise NotImplementedError(f"Unknown loss function name: {loss_name}")


class EnergyPerAtomMAELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, 
                input: torch.Tensor, 
                target: torch.Tensor, 
                natoms: torch.Tensor,
    ):
        # batch size
        assert input.shape[0] == natoms.shape[0] # (batch size,) # ex) [48, 24, 48, 96] for BS=4
        
        abs_error = torch.abs((target - input) / natoms)
        if self.reduction == "mean":
            return torch.mean(abs_error)
        elif self.reduction == "sum":
            return torch.sum(abs_error)


class EnergyPerAtomMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, 
                input: torch.Tensor, 
                target: torch.Tensor, 
                natoms: torch.Tensor,
    ):
        # batch size
        assert input.shape[0] == natoms.shape[0] # (batch size,) # ex) [48, 24, 48, 96] for BS=4
        
        squared_error = torch.square((target - input) / natoms)
        if self.reduction == "mean":
            return torch.mean(squared_error)
        elif self.reduction == "sum":
            return torch.sum(squared_error)


class DDPLoss(nn.Module):
    def __init__(self, loss_fn, reduction="mean"):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "sum"
        self.reduction = reduction
        assert reduction in ["mean", "sum", "mean_over_dim"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor = None,
        batch_size: int = None,
    ):
        # zero out nans, if any
        found_nans_or_infs = not torch.all(input.isfinite())
        if found_nans_or_infs is True:
            logging.warning("Found nans while computing loss")
            input = torch.nan_to_num(input, nan=0.0)

        if natoms is None:
            loss = self.loss_fn(input, target)
        else:  # atom-wise loss
            loss = self.loss_fn(input, target, natoms)
        if self.reduction in ["mean", "mean_over_dim"]:
            num_samples = (
                batch_size if batch_size is not None else input.shape[0]
            )
            num_samples = distutils.all_reduce(
                num_samples, device=input.device
            )
            if self.reduction == "mean_over_dim":
                # In DDPLoss in OCP (just using "mean"), 
                # force loss are averaged over num of atoms (i.e., dimension is not considered).
                # But, some codes like NequIP and SIMPLE-NN is implemented so that force loss is averaged by (3 * num of atoms)
                # So, we deal with this loss property
                num_samples *= 3 

            # Multiply by world size since gradients are averaged
            # across DDP replicas
            return loss * distutils.get_world_size() / num_samples
        else:
            return loss
