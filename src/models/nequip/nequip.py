"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import torch
import inspect

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel

from nequip.utils.config import Config
from nequip.data import AtomicDataDict, AtomicData
from nequip.data.transforms import TypeMapper
from nequip.model import SimpleIrrepsConfig, ForceOutput, PartialForceOutput, StressForceOutput

from src.common.utils import bm_logging
from src.models.nequip.energy_model import EnergyModel
from src.models.nequip.rescale import RescaleEnergyEtc, PerSpeciesRescale
from src.models.nequip.utils import (
    compute_avg_num_neighbors, 
    compute_global_shift_and_scale,
    compute_per_species_shift_and_scale
)


def set_model_config_based_on_data_statistics(model_config, type_mapper, dataset_name, use_scale_shift=True, initialize=True):
    # add statistics results to config
    dataset = None
    if initialize and dataset_name is not None:
        dataset_class = registry.get_dataset_class("lmdb")
        dataset = dataset_class({"src": dataset_name})

    # 1) avg_num_neighbors (required by EnergyModel)
    avg_num_neighbors = model_config.get("avg_num_neighbors", None)
    if initialize and avg_num_neighbors == "auto":
        assert dataset is not None
        avg_num_neighbors = compute_avg_num_neighbors(
            config=model_config, 
            initialize=initialize, 
            dataset=dataset,
            transform=type_mapper,
        )
        
    model_config["avg_num_neighbors"] = avg_num_neighbors
    bm_logging.info(f"avg_num_neighbors used in interaction layers is {avg_num_neighbors}")

    # 2) per_species_rescale_shifts and per_species_rescale_scales (required by PerSpeciesRescale)
    if "PerSpeciesRescale" in model_config["model_builders"]:
        if use_scale_shift:
            shifts, scales, arguments_in_dataset_units = compute_per_species_shift_and_scale(
                config=model_config, 
                initialize=initialize, 
                dataset=dataset,
                transform=type_mapper,
            )
        else:
            shifts = None
            scales = None
            arguments_in_dataset_units = False
            bm_logging.info("[per species rescale] Scales and shifts are not used")
        model_config["per_species_rescale_shifts"] = shifts
        model_config["per_species_rescale_scales"] = scales
        model_config["arguments_in_dataset_units"] = arguments_in_dataset_units

    # 3) global_rescale_shift and global_rescale_scale (required by RescaleEnergyEtc (i.e., GlobalRescale))
    if "RescaleEnergyEtc" in model_config["model_builders"]:
        if use_scale_shift:
            global_shift, global_scale = compute_global_shift_and_scale(
                config=model_config, 
                initialize=initialize, 
                dataset=dataset,
                transform=type_mapper,
            )
        else:
            global_shift = None
            global_scale = None
            bm_logging.info("[global rescale] Scale and shift are not used")
        model_config["global_rescale_shift"] = global_shift
        model_config["global_rescale_scale"] = global_scale

    if dataset is not None:
        dataset.close_db()
    return model_config


def initiate_model_by_builders(builders, config, initialize):
    model = None
    for builder in builders:
        pnames = inspect.signature(builder).parameters
        params = {}
        if "config" in pnames:
            params["config"] = config

        if "initialize" in pnames:
            params["initialize"] = initialize

        if "model" in pnames:
            if model is None:
                raise RuntimeError(f"Builder {builder.__name__} asked for the model as an input, but no previous builder has returned a model")
            params["model"] = model
        else:
            if model:
                raise RuntimeError(f"All model_builders after the first one that returns a model must take the model as an argument; {builder.__name__} doesn't")
        model = builder(**params)
    return model

@registry.register_model("nequip")
class NequIPWrap(BaseModel):
    def __init__(
        self,
        num_atoms, # not used
        bond_feat_dim, # not used
        num_targets,
        cutoff=6.0,
        max_neighbors=None, 
        use_pbc=True,
        regress_forces=True,
        regress_stress=False,
        otf_graph=False,
        # data-related arguments (type mapper and statistics)
        num_types=None,
        type_names=None,
        chemical_symbol_to_type=None,
        chemical_symbols=None,
        dataset=None, # train dataset path
        # architecture arguments
        model_builders=[
            "SimpleIrrepsConfig",
            "EnergyModel",
            "PerSpeciesRescale",
            "ForceOutput",
            "RescaleEnergyEtc",
        ],
        num_layers=3,
        l_max=1,
        parity=True,
        num_features=32,
        nonlinearity_type="gate",
        resnet=False,
        nonlinearity_scalars={"e": "silu", "o": "tanh"},
        nonlinearity_gates={"e": "silu", "o": "tanh"},
        num_basis=8,
        BesselBasis_trainable=True,
        PolynomialCutoff_p=6,
        invariant_layers=2,
        invariant_neurons=64,
        use_sc=True,
        avg_num_neighbors=None,
        per_species_rescale_shifts_trainable=False,
        per_species_rescale_scales_trainable=False,
        per_species_rescale_shifts="dataset_per_atom_total_energy_mean",
        per_species_rescale_scales="dataset_forces_rms",
        global_rescale_shift_trainable=False,
        global_rescale_scale_trainable=False,
        global_rescale_shift=None,
        global_rescale_scale="dataset_forces_rms",
        # normalization on/off
        use_scale_shift=True,
        # initialize: False = load checkpoint, True = data seeing is the first
        initialize=True,
    ):
        self.num_targets = num_targets
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.regress_stress = regress_stress
        self.otf_graph = otf_graph
        self.cutoff = cutoff

        if self.otf_graph:
            raise NotImplementedError("on-the-fly garph generation is not enabled for NequIP")
        
        self.max_neighbors = max_neighbors
        super().__init__()

        model_config = dict(
            model_builders=model_builders,
            r_max=self.cutoff,
            num_layers=num_layers,
            l_max=l_max,
            parity=parity,
            num_features=num_features,
            nonlinearity_type=nonlinearity_type,
            resnet=resnet,
            nonlinearity_scalars=nonlinearity_scalars,
            nonlinearity_gates=nonlinearity_gates,
            num_basis=num_basis,
            BesselBasis_trainable=BesselBasis_trainable,
            PolynomialCutoff_p=PolynomialCutoff_p,
            invariant_layers=invariant_layers,
            invariant_neurons=invariant_neurons,
            use_sc=use_sc,
            avg_num_neighbors=avg_num_neighbors,
            ## belows are default
            per_species_rescale_shifts_trainable=per_species_rescale_shifts_trainable,
            per_species_rescale_scales_trainable=per_species_rescale_scales_trainable,
            per_species_rescale_shifts=per_species_rescale_shifts,
            per_species_rescale_scales=per_species_rescale_scales,
            global_rescale_shift_trainable=global_rescale_shift_trainable,
            global_rescale_scale_trainable=global_rescale_scale_trainable,
            global_rescale_shift=global_rescale_shift,
            global_rescale_scale=global_rescale_scale,
            dataset_statistics_stride=1,
        )
        model_config = Config.from_dict(model_config)

        # AtomicDataset includes TypeMapper. 
        # To preprocess atomic number into type, 
        # it is required to use TypeMapper as a transform in the collater function.
        self.type_mapper = TypeMapper(
            type_names=type_names,
            chemical_symbol_to_type=chemical_symbol_to_type,
            chemical_symbols=chemical_symbols,
        )
        if num_types is not None:
            assert (num_types == self.type_mapper.num_types), "inconsistant config & dataset"
        if type_names is not None:
            assert (type_names == self.type_mapper.type_names), "inconsistant config & dataset"
        model_config["num_types"] = self.type_mapper.num_types
        model_config["type_names"] = self.type_mapper.type_names

        # compute statistics (similar to Normalizers of OCP)
        # or load the pre-computed values
        self.use_scale_shift = use_scale_shift
        model_config = set_model_config_based_on_data_statistics(
            model_config=model_config, 
            type_mapper=self.type_mapper, 
            dataset_name=dataset,
            use_scale_shift=use_scale_shift,
            initialize=initialize,
        )

        self.avg_num_neighbors = model_config["avg_num_neighbors"]

        # constrcut the NequIP model
        builders = [eval(module) for module in model_config["model_builders"]]
        self.nequip_model = initiate_model_by_builders(
            builders=builders, 
            config=model_config, 
            initialize=initialize,
        )

        # maintain rescale layers individually
        self.rescale_layers = []
        outer_layer = self.nequip_model
        while hasattr(outer_layer, "unscale"):
            self.rescale_layers.append(outer_layer)
            outer_layer = getattr(outer_layer, "model", None)

    def do_unscale(self, data, force_process=False):
        if not self.use_scale_shift:
            return data
            
        # unscaling (by RescaleEnergyEtc, or GlobalRescale)
        # : (x - shift) / scale
        for layer in self.rescale_layers:
            data = layer.unscale(data, force_process=force_process)
        return data

    def do_scale(self, data, force_process=False):
        if not self.use_scale_shift:
            return data

        # scaling (by RescaleEnergyEtc, or GlobalRescale)
        # : x * scale + shift
        for layer in self.rescale_layers[::-1]:
            data = layer.scale(data, force_process=force_process)
        return data

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # data is already moved to device by OCPDataParallel (ocp/ocpmodels/common/data_parallel.py)
        input_data = AtomicData.to_AtomicDataDict(data)

        # model forward
        out = self.nequip_model(input_data)
        
        # return values required in an OCP-based trainer
        if self.regress_stress:
            return out[AtomicDataDict.TOTAL_ENERGY_KEY], out[AtomicDataDict.FORCE_KEY], out[AtomicDataDict.STRESS_KEY]
        elif self.regress_forces:
            return out[AtomicDataDict.TOTAL_ENERGY_KEY], out[AtomicDataDict.FORCE_KEY]
        else:
            return out[AtomicDataDict.TOTAL_ENERGY_KEY]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.nequip_model.parameters())
