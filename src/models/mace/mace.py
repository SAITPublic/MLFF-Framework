"""
written by byunggook.na (SAIT)
"""

import numpy as np
import torch
from e3nn import o3
import ase

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.datasets import LmdbDataset 

# pre-defined modules in MACE
from mace.modules import ScaleShiftMACE
from mace.modules import interaction_classes, gate_dict
from mace.tools import get_atomic_number_table_from_zs

from src.common.utils import bm_logging # benchmark logging


@registry.register_model("mace")
class MACEWrap(BaseModel):
    def __init__(
        self,
        num_atoms, # not used
        bond_feat_dim, # not used
        num_targets,
        cutoff=5.0,
        max_neighbors=None, # not used?
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        # data-related arguments (which are obtained on-the-fly)
        chemical_symbols=None,
        z_table=None,
        atomic_energies=None, # list, which will be converted into np.array
        avg_num_neighbors=1,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        # architecture arguments
        num_radial_basis=8,
        num_cutoff_basis=5,
        interaction="RealAgnosticResidualInteractionBlock",
        interaction_first="RealAgnosticResidualInteractionBlock",
        max_ell=3,
        correlation=3,
        num_interactions=2,
        MLP_irreps="16x0e",
        hidden_irreps="32x0e",
        gate="silu", # silu, tanh, abs, None
        **kwargs,
    ):
        self.num_targets = num_targets
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.cutoff = cutoff

        if self.otf_graph:
            raise NotImplementedError("on-the-fly garph generation is not enabled for MACE")

        self.max_neighbors = max_neighbors
        super().__init__()

        # data-related arguments 
        # (which should be loaded when using the model as a calculator)
        # TODO: check the loading!
        assert (z_table is not None or chemical_symbols is not None)
        if chemical_symbols is not None:
            zs = [ase.atom.atomic_numbers[atom] for atom in chemical_symbols]
            self.z_table = get_atomic_number_table_from_zs(zs)
        else:
            if z_table is not None:
                self.z_table = z_table

        assert atomic_energies is not None
        self.atomic_energies = np.array(atomic_energies)
        self.avg_num_neighbors = avg_num_neighbors
        self.hidden_irreps = hidden_irreps

        # build a model
        #  For now (2023. 3. 28), we assume that only MACE or ScaleShiftMACE is used
        #  MACE : ScaleShiftMACE with scaling option = no_scaling
        #  ScaleShiftMACE : ScaleShiftMACE with scaling option = std_scaling or rms_forces_scaling (defealt)
        # 1) prepare a model config (including hidden_irreps)
        model_config = dict(
            r_max=self.cutoff,
            num_bessel=num_radial_basis,
            num_polynomial_cutoff=num_cutoff_basis,
            max_ell=max_ell,
            interaction_cls=interaction_classes[interaction],
            num_interactions=num_interactions,
            num_elements=len(self.z_table),
            hidden_irreps=o3.Irreps(self.hidden_irreps),
            atomic_energies=self.atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=self.z_table.zs,
            correlation=correlation,
            gate=gate_dict[gate],
            MLP_irreps=o3.Irreps(MLP_irreps),
            # arguments distingushing MACE and ScaleShiftMACE
            atomic_inter_scale=atomic_inter_scale,
            atomic_inter_shift=atomic_inter_shift,
            interaction_cls_first=interaction_classes[interaction_first],
        )
        
        # 2) initiate a model
        self.mace_model = ScaleShiftMACE(**model_config)

        # 3) set flags for which outputs are computed
        self.compute_force = regress_forces
        self.compute_virials = False
        self.compute_stress = False

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # data is already moved to device 
        # by OCPDataParallel (ocpmodels/common/data_parallel.py)

        # model forward
        # : output is dict
        out = self.mace_model(data, training=self.training)
        
        # return values required in an OCP-based trainer
        if self.regress_forces:
            return out["energy"], out["forces"]
        else:
            return out["energy"]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.mace_model.parameters())