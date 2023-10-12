"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import time
from collections import OrderedDict

import torch
from torch_geometric.data import Data, Batch

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_state_dict
from ocpmodels.modules.normalizer import Normalizer

from nequip.data import AtomicData, AtomicDataDict
from nequip.utils.torch_geometric.batch import Batch as BatchNequIP

from mace import data as mace_data
from mace.tools.torch_geometric.batch import Batch as BatchMACE

from src.common.collaters.parallel_collater_nequip import convert_ocp_Data_into_nequip_AtomicData
from src.common.collaters.parallel_collater_mace import convert_ocp_Data_into_mace_AtomicData
from src.common.utils import bm_logging
from src.modules.normalizer import NormalizerPerAtom, log_and_check_normalizers
from src.preprocessing.atoms_to_graphs import AtomsToGraphsWithTolerance


class BenchmarkCalculator(Calculator):
    
    implemented_properties = ["energy", "forces"]
    
    def __init__(self, ckpt=None, device=torch.device("cpu"), **kwargs):
        Calculator.__init__(self, **kwargs)
        self.device = device

        assert ckpt is not None
        ckpt_config = ckpt["config"]
        self.model_name = ckpt_config["model_name"]

        # adjust the model-specific attributes for evaluation mode
        if self.model_name in ["nequip", "allegro"]:
            ckpt_config["model_attributes"]["initialize"] = False
            if "data_normalization" in ckpt_config["model_attributes"].keys():
                bm_logging.info(f"This configuration is old-styled. `data_normalization` is converted into `use_scale_shift` ({ckpt_config['model_attributes']['data_normalization']} in this config file).")
                ckpt_config["model_attributes"]["use_scale_shift"] = ckpt_config["model_attributes"]["data_normalization"]
                del ckpt_config["model_attributes"]["data_normalization"]
        elif self.model_name in ["bpnn"]:
            ckpt_config["model_attributes"]["pca_path"] = None 
            ckpt_config["model_attributes"]["dataset_path"] = None

        # construct a model in the ckpt
        model_class = registry.get_model_class(self.model_name)
        self.model = model_class(
            num_atoms = None, # not used
            bond_feat_dim = None, # not used
            num_targets = 1, # always 1 (for energy)
            **ckpt_config["model_attributes"],
        )
        bm_logging.info(f"Set a calculator based on {self.model_name} class (direct force prediction: {ckpt_config['model_attributes'].get('direct_forces', False)})")

        # load the trained parameters of the model (and move it to GPU)
        model_state_dict = OrderedDict()
        for key, val in ckpt["state_dict"].items():
            k = key
            while k.startswith("module."):
                k = k[7:]
            model_state_dict[k] = val
        load_state_dict(module=self.model, state_dict=model_state_dict, strict=True)

        # load auxiliary tensors for some models
        if self.model_name in ["bpnn"]:
            self.model.pca = ckpt["pca"]

        # move the model in GPU
        self.model = self.model.to(self.device)

        # evaluation mode
        self.model.eval() 

        # set normalizers (if it exists)
        if ckpt_config.get("data_config_style", "OCP") == "OCP":
            # OCP data config style
            normalizer = ckpt_config["dataset"]
        else:
            # SAIT data config style
            assert "normalizer" in ckpt_config.keys()
            normalizer = ckpt_config["normalizer"]
        self.normalizers = {}
        if normalizer.get("normalize_labels", False):
            self.normalization_per_atom = normalizer.get("per_atom", False)
            if self.normalization_per_atom:
                self.normalizers["target"] = NormalizerPerAtom(mean=0.0, std=1.0, device=self.device)
            else:
                self.normalizers["target"] = Normalizer(mean=0.0, std=1.0, device=self.device)
            self.normalizers["target"].load_state_dict(ckpt["normalizers"]["target"])
            self.normalizers["grad_target"] = Normalizer(mean=0.0, std=1.0, device=self.device)
            self.normalizers["grad_target"].load_state_dict(ckpt["normalizers"]["grad_target"])
            log_and_check_normalizers(self.normalizers["target"], self.normalizers["grad_target"], loaded=True)
        
        # extract arguments required to convert atom data into graph ocp data
        self.cutoff = self.model.cutoff
        self.max_neighbors = self.model.max_neighbors
        self.pbc = torch.tensor([self.model.use_pbc] * 3)

        # set data converter
        # 1) ase.Atoms -> torch_geometric (pyg) Data structure
        self.atoms_to_pyg_data = AtomsToGraphsWithTolerance(
            max_neigh=self.max_neighbors,
            radius=self.cutoff,
            r_energy=False,
            r_forces=False,
            r_fixed=True,
            r_distances=False,
            r_pbc=self.model.use_pbc,
            r_edges=(not self.model.otf_graph), # if model does not support on-the-fly edge generation, the data converter needs to get edges.
            tolerance=1e-8,
        )

        # 2) pyg Data structure -> a data structure defined in each model
        if self.model_name in ["nequip", "allegro"]:
            self.convert_atoms_to_batch = self.convert_atoms_to_nequip_batch
        elif self.model_name in ["mace"]:
            self.convert_atoms_to_batch = self.convert_atoms_to_mace_batch
        else:
            self.convert_atoms_to_batch = self.convert_atoms_to_ocp_batch

    def convert_atoms_to_ocp_batch(self, atoms):
        # convert ase.Atoms into pytorch_geometric data
        data = self.atoms_to_pyg_data.convert(atoms)
        batch = Batch.from_data_list([data]) # batch size = 1
        if not self.model.otf_graph:
            batch.neighbors = torch.tensor([data.edge_index.shape[1]])
        return batch

    def convert_atoms_to_nequip_batch(self, atoms):
        # convert ase.Atoms into AtomicData of NequIP
        data = self.atoms_to_pyg_data.convert(atoms)
        data = convert_ocp_Data_into_nequip_AtomicData(data, self.model.type_mapper)
        batch = BatchNequIP.from_data_list([data]) # batch size = 1
        return batch

    def convert_atoms_to_mace_batch(self, atoms):
        # convert ase.Atoms into AtomicData of MACE
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
                if self.normalization_per_atom:
                    # Inference is performed on a single snapshot
                    # Thus, for N in denorm(), we use forces.shape[0] which is num of atoms of this snapshot.
                    energy = self.normalizers["target"].denorm(energy, forces.shape[0])
                else:
                    energy = self.normalizers["target"].denorm(energy)
                forces = self.normalizers["grad_target"].denorm(forces)
            return energy, forces

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes, measure_time=False):
        """
        atoms: ase.Atoms
        """
        # set atoms attribute
        Calculator.calculate(self, atoms=atoms, properties=properties, system_changes=system_changes)

        # convert ase.Atoms into a data batch format compaitible with MLFF models
        t1 = time.time()
        batch = self.convert_atoms_to_batch(atoms)
        batch = batch.to(self.device)
        
        # MLFF model inference
        t2 = time.time()
        energy, forces = self.model(batch)

        # de-normalization (if it was used during training the model)
        energy, forces = self.denormalization(energy, forces)
        t3 = time.time()
        
        # save the results
        self.results = {
            "energy": energy.item(),
            "forces": forces.detach().cpu().numpy(),
        }

        if measure_time:
            self.time_data_preparation = t2-t1
            self.time_model_inference = t3-t2