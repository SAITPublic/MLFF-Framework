from collections import OrderedDict

import torch
from torch_geometric.data import Data, Batch

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_state_dict
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.preprocessing import AtomsToGraphs

from nequip.data import AtomicData, AtomicDataDict
from nequip.utils.torch_geometric.batch import Batch as BatchNequIP

from mace import data as mace_data
from mace.tools.torch_geometric.batch import Batch as BatchMACE

from src.common.collaters.parallel_collater_nequip import convert_ocp_Data_into_nequip_AtomicData
from src.common.collaters.parallel_collater_mace import convert_ocp_Data_into_mace_AtomicData
from src.common.utils import bm_logging # benchmark logging

class BenchmarkCalculator(Calculator):
    def __init__(self, ckpt=None, device=torch.device("cpu"), **kwargs):
        Calculator.__init__(self, **kwargs)

        assert ckpt is not None

        ckpt_config = ckpt["config"]
        self.device = device

        # construct a model in the ckpt
        self.model_name = ckpt_config["model_name"]
        model_class = registry.get_model_class(self.model_name)
        self.model = model_class(
            num_atoms = None, # not used
            bond_feat_dim = None, # not used
            num_targets = 1, # always 1
            **ckpt_config["model_attributes"],
        )
        bm_logging.info(f"Set a calculator based on {self.model_name} class")

        # load the trained parameters of the model (and move it to GPU)
        model_state_dict = OrderedDict()
        for key, val in ckpt["state_dict"].items():
            k = key
            while k.startswith("module."):
                k = k[7:]
            model_state_dict[k] = val
        load_state_dict(module=self.model, state_dict=model_state_dict, strict=True)
        self.model = self.model.to(self.device)

        # evaluation mode
        self.model.eval() 
        # because the input snapshots are new, we should use on-the-fly graph generation
        self.model.otf_graph = True 

        # set normalizers (if it exists)
        self.normalizers = {}
        if ckpt_config["dataset"].get("normalize_labels", False):
            for key in ["target", "grad_target"]:
                self.normalizers[key] = Normalizer(mean=0.0, std=1.0, device=self.device)
                self.normalizers[key].load_state_dict(ckpt["normalizers"][key])
            bm_logging.info(f"Loaded normalizers")
            bm_logging.info(f" - energy : shift ({self.normalizers['target'].mean}) scale ({self.normalizers['target'].std})")
            bm_logging.info(f" - forces : shift ({self.normalizers['grad_target'].mean}) scale ({self.normalizers['grad_target'].std})")
        
        # extract arguments required to convert atom data into graph ocp data
        self.cutoff = self.model.cutoff
        self.max_neighbors = self.model.max_neighbors
        self.pbc = torch.tensor([self.model.use_pbc] * 3)

        # set data converter
        # 1) ase.Atoms -> torch_geometric (pyg) Data structure
        self.atoms_to_pyg_data = AtomsToGraphs(
            max_neigh=self.max_neighbors,
            radius=self.cutoff,
            r_energy=True,
            r_forces=True,
            r_fixed=True,
            r_distances=False,
            r_pbc=self.model.use_pbc,
            r_edges=True, ## for on-the-fly
        )

        # 2) pyg Data structure -> a data structure defined in each model
        if self.model_name in ["nequip", "allegro"]:
            self.convert_atoms_to_batch = self.convert_atoms_to_nequip_batch
        elif self.model_name in ["mace"]:
            self.convert_atoms_to_batch = self.convert_atoms_to_mace_batch
        else:
            self.convert_atoms_to_batch = self.convert_atoms_to_ocp_batch

    def convert_atoms_to_ocp_batch(self, atoms):
        # convert atoms into pytorch_geometric data
        # : otf = True, which means the atoms are converted into graph in model forward() on-the-fly
        data = self.atoms_to_pyg_data.convert(atoms)
        batch = Batch.from_data_list([data]) # batch size = 1
        return batch

    def convert_atoms_to_nequip_batch(self, atoms):
        # convert atoms into AtomicData of NequIP
        # : When constructing AtomicData, the atoms are converted into graph
        # data = AtomicData.from_ase(atoms=atoms, r_max=self.cutoff)
        # # remove labels
        # for k in AtomicDataDict.ALL_ENERGY_KEYS:
        #     if k in data:
        #         del data[k]
        # data = self.model.type_mapper(data)
        # data.pbc = self.pbc
        # batch = BatchNequIP.from_data_list([data]) # batch size = 1
        # batch = batch.to(self.device)
        # return batch
        data = self.atoms_to_pyg_data.convert(atoms)
        data = convert_ocp_Data_into_nequip_AtomicData(data, self.model.type_mapper)
        batch = BatchNequIP.from_data_list([data]) # batch size = 1
        return batch

    def convert_atoms_to_mace_batch(self, atoms):
        # convert atoms into AtomicData of MACE
        # : When constructing mace_data.AtomicData, the atoms are converted into graph
        # config = mace_data.config_from_atoms(atoms)
        # data = mace_data.AtomicData.from_config(config, z_table=self.model.z_table, cutoff=self.cutoff)
        # data.pbc = self.pbc
        # batch = BatchMACE.from_data_list([data]) # batch size = 1
        # return batch
        data = self.atoms_to_pyg_data.convert(atoms)
        data = convert_ocp_Data_into_mace_AtomicData(data, self.model.z_table)
        batch = BatchMACE.from_data_list([data]) # batch size = 1
        return batch

    def denormalization(self, energy, forces):
        if self.model_name in ["nequip", "allegro", "mace"]:
            # the model output is already in real unit in evaluation mode
            return energy, forces
        else:
            if len(self.normalizers) > 0:
                # if normalization is used in the trainer, model output values should be de-normalized.
                # (because ground truth trained by a model was normalized)
                energy = self.normalizers["target"].denorm(energy)
                forces = self.normalizers["grad_target"].denorm(forces)
            return energy, forces

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        """
        atoms: ase.Atoms
        """
        # set atoms attribute
        Calculator.calculate(self, atoms=atoms, properties=properties, system_changes=system_changes)

        # convert atoms (ASE) into a data batch format compaitible with MLFF models
        batch = self.convert_atoms_to_batch(atoms)
        batch = batch.to(self.device)
        
        # MLFF model inference
        energy, forces = self.model(batch)

        # de-normalization (if it is used)
        energy, forces = self.denormalization(energy, forces)
        
        # save the results
        self.results = {
            "energy": energy.item(),
            "forces": forces.detach().cpu().numpy(),
        }