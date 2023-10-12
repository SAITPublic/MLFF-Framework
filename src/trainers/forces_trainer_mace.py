"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import ast
import torch
import numpy as np
from pathlib import Path

import ase

from ocpmodels.common.registry import registry

from mace.tools import get_atomic_number_table_from_zs

from src.common.utils import bm_logging
from src.common.collaters.parallel_collater_mace import ParallelCollaterMACE
from src.trainers.forces_trainer import ForcesTrainer
from src.models.mace.utils import (
    compute_average_E0s, 
    compute_avg_num_neighbors,
    compute_mean_std_atomic_inter_energy,
    compute_mean_rms_energy_forces,
)


@registry.register_trainer("forces_mace")
class MACEForcesTrainer(ForcesTrainer):
    """
    Trainer class for the S2EF (Structure to Energy & Force) task, 
    and this class is especially used to train MACE models.
    """
    def __init__(self, config):
        super().__init__(config)

        # NOTE: all configurations have equal weights in this benchmark
        # config_type_weights = {"Default": 1.0}

        # TODO: SWA
        # In MACE code, there is SWA option (stochastic weight averaging).
    
    def _parse_config(self, config):
        trainer_config = super()._parse_config(config)

        # MACE does not need OCP normalizer (it uses own normaliation strategy)
        ocp_normalize_flag = False
        data_config_style = trainer_config.get("data_config_style", "OCP")
        if data_config_style == "OCP":
            ocp_normalize_flag = trainer_config["dataset"].get("normalize_labels", False)
            trainer_config["dataset"]["normalize_labels"] = False
        elif data_config_style == "SAIT":
            ocp_normalize_flag = trainer_config["normalizer"].get("normalize_labels", False)
            trainer_config["normalizer"]["normalize_labels"] = False
        if ocp_normalize_flag:
            bm_logging.info("In the given configuration file or the configuration saved in the checkpoint, `normalize_labels` is set as `True` ")
            bm_logging.info("  MACE does not need OCP normalizers, instead it uses own normalization strategy.")
            bm_logging.info("  Hence `normalize_labels` will be changed as `False` to turn off the OCP normalizer operation.")
            bm_logging.info("  You can control their own normalization strategy by changing `scaling` and `shifting` in the model configuration.")
        
        # set hidden irreps
        hidden_irreps = trainer_config["model_attributes"].get("hidden_irreps", None)
        num_channels = trainer_config["model_attributes"].get("num_channels", None)
        max_L = trainer_config["model_attributes"].get("max_L", None)
        if num_channels is not None and max_L is not None:
            assert num_channels > 0, "num_channels must be positive integer"
            assert (
                max_L >= 0 and max_L < 4
            ), "max_L must be between 0 and 3, if you want to use larger specify it via the hidden_irreps keyword"
            hidden_irreps = f"{num_channels:d}x0e"
            if max_L > 0:
                hidden_irreps += f" + {num_channels:d}x1o"
            if max_L > 1:
                hidden_irreps += f" + {num_channels:d}x2e"
            if max_L > 2:
                hidden_irreps += f" + {num_channels:d}x3o"
        trainer_config["model_attributes"]["hidden_irreps"] = hidden_irreps
        bm_logging.info(f"Hidden irreps: {hidden_irreps}")
        
        # set interaction_first
        if not trainer_config["model_attributes"].get("shifting", False):
            # MACE : fix the first interaction class
            if ("interaction_first" in trainer_config["model_attributes"] and
                trainer_config["model_attributes"]["interaction_first"] != "RealAgnosticInteractionBlock"
            ):
                bm_logging.warning(f"When `shifting` is set as False, RealAgnosticInteractionBlock is forcely assigned as `interaction_first`")
            trainer_config["model_attributes"]["interaction_first"] = "RealAgnosticInteractionBlock"

        # set z_table
        zs = trainer_config["model_attributes"].get("chemical_symbols", None)
        if zs is not None:
            # if atom species are given
            zs = [ase.atom.atomic_numbers[atom] for atom in zs]
            self.z_table = get_atomic_number_table_from_zs(zs)
        else:
            # otherwise, we get atom species on-the-fly
            dataset_class = registry.get_dataset_class(trainer_config["task"].get("dataset", "lmdb"))
            datasets = []
            if trainer_config.get("dataset", None):
                datasets.append(dataset_class(trainer_config["dataset"]))
            if trainer_config.get("val_dataset", None):
                datasets.append(dataset_class(trainer_config["val_dataset"]))
            self.z_table = get_atomic_number_table_from_zs(
                int(z.item())
                for dataset in datasets
                for data in dataset
                for z in data.atomic_numbers
            )
            for dataset in datasets:
                dataset.close_db()
            trainer_config["model_attributes"]["z_table"] = self.z_table
        bm_logging.info(f"z_table : {self.z_table}")
        
        return trainer_config
    
    def initiate_collater(self):
        return ParallelCollaterMACE(
            num_gpus=0 if self.cpu else 1,
            otf_graph=self.config["model_attributes"].get("otf_graph", False),
            use_pbc=self.config["model_attributes"].get("use_pbc", False),
            z_table=self.z_table,
        )

    def _do_data_related_settings(self):
        """ After setting dataset and loader, this function is called."""
        # load the precomputed results (if they exist)
        data_config_style = self.config.get("data_config_style", "OCP")
        if data_config_style == "OCP":
            # OCP data config style
            mace_statistics_file_path = (Path(self.config['dataset']['src']).parent / "MACE_statistics.pt")
        if data_config_style == "SAIT":
            # SAIT data config style
            assert isinstance(self.config["dataset"], list)
            if len(self.config["dataset"]) > 1:
                bm_logging("The first source of training datasets will be used to obtain atomic_energies, avg_num_neighbors, atomic_inter_scale, and atomic_inter_shift.")
            if not Path(self.config["dataset"][0]["src"]).exists():
                raise RuntimeError("The lmdb source file or directory should be specified.")
            mace_statistics_file_path = (Path(self.config['dataset'][0]['src']).parent / "MACE_statistics.pt")

        mace_statistics = {}
        if mace_statistics_file_path.is_file():
            mace_statistics = torch.load(mace_statistics_file_path)
            bm_logging.info(f"Load the computed results from {mace_statistics_file_path}")

        # 2. compute atomic energies (on train dataset)
        # : assume single lmdb
        atomic_energies_dict = {}
        if self.config["model_attributes"].get("E0s", "average").lower() == "average":
            if "atomic_energies" not in mace_statistics:
                bm_logging.info("Computing average Atomic Energies using least squares regression")
                atomic_energies_dict = compute_average_E0s(
                    train_dataset=self.train_dataset, 
                    z_table=self.z_table,
                )
                atomic_energies = [atomic_energies_dict[z] for z in self.z_table.zs]
                mace_statistics["atomic_energies"] = atomic_energies
            else:
                atomic_energies = mace_statistics["atomic_energies"]
        else:
            try:
                atomic_energies_dict = ast.literal_eval(args.E0s)
                assert isinstance(atomic_energies_dict, dict)
                atomic_energies = [atomic_energies_dict[z] for z in self.z_table.zs]
            except Exception as e:
                raise RuntimeError(
                    f"E0s specified invalidly, error {e} occured"
                ) from e
        self.atomic_energies = atomic_energies
        self.config["model_attributes"]["atomic_energies"] = atomic_energies
        bm_logging.info(f"Atomic energies: {self.atomic_energies}")

        # 3. compute average num of neighbors (on train dataset)
        avg_num_neighbors = self.config["model_attributes"].get("avg_num_neighbors", 1)
        if self.config["model_attributes"].get("compute_avg_num_neighbors", True):
            if "avg_num_neighbors" not in mace_statistics:
                avg_num_neighbors = compute_avg_num_neighbors(
                    data_loader=self.train_loader
                )
                mace_statistics["avg_num_neighbors"] = avg_num_neighbors
            else:
                avg_num_neighbors = mace_statistics["avg_num_neighbors"]
        self.config["model_attributes"]["avg_num_neighbors"] = avg_num_neighbors
        bm_logging.info(f"Average number of neighbors: {avg_num_neighbors}")

        # 4. set scaling and shifting
        scaling = self.config["model_attributes"].get("scaling", "rms_forces_scaling")
        if "scaling_type" not in mace_statistics:
            if scaling == "no_scaling":
                mean, std = 0.0, 1.0
                bm_logging.info("No scaling (and neither shifting) selected")
            elif scaling == "std_scaling":
                mean, std = compute_mean_std_atomic_inter_energy(
                    data_loader=self.train_loader,
                    atomic_energies=np.array(self.atomic_energies),
                )
            elif scaling == "rms_forces_scaling":
                mean, std = compute_mean_rms_energy_forces(
                    data_loader=self.train_loader,
                    atomic_energies=np.array(self.atomic_energies),
                )
            else:
                raise RuntimeError(f"{scaling} is not supported")
            mace_statistics["scaling_type"] = scaling
            mace_statistics["mean"] = mean
            mace_statistics["std"] = std
        else:
            if scaling != mace_statistics["scaling_type"]:
                raise RuntimeError(f"There is difference between `scaling_type` in this training config and `scaling_type` saved in {mace_statistics_file_path}")
            mean = mace_statistics["mean"]
            std = mace_statistics["std"]

        if not self.config["model_attributes"].get("shifting", False):
            # MACE : no shifting
            self.config["model_attributes"]["atomic_inter_scale"] = std
            self.config["model_attributes"]["atomic_inter_shift"] = 0.0
        else:
            # ScaleShiftMACE
            self.config["model_attributes"]["atomic_inter_scale"] = std
            self.config["model_attributes"]["atomic_inter_shift"] = mean

        # save the precomputed results
        if not mace_statistics_file_path.exists():
            torch.save(mace_statistics, mace_statistics_file_path)
            bm_logging.info(f"Save the computed results at {mace_statistics_file_path}")

    def _split_trainable_params_optimizer_weight_decay(self):
        # as in mace code
        params_decay = []
        params_no_decay = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "embedding" in name:
                    params_no_decay += [param]
                elif "interactions" in name:
                    if "linear.weight" in name or "skip_tp_full.weight" in name:
                        params_decay += [param]
                    else:
                        params_no_decay += [param]
                elif "readouts" in name:
                    params_no_decay += [param]
                else:
                    params_decay += [param]
        return params_decay, params_no_decay

    def _compute_loss(self, out, batch_list):
        for batch in batch_list:
            batch.y = batch.energy
            batch.force = batch.forces
            batch.natoms = torch.bincount(batch.batch)
        return super()._compute_loss(out=out, batch_list=batch_list)

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        for batch in batch_list:
            batch.y = batch.energy
            batch.force = batch.forces
            batch.natoms = torch.bincount(batch.batch)
        return super()._compute_metrics(out=out, batch_list=batch_list, evaluator=evaluator, metrics=metrics)    