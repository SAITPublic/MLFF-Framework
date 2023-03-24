"""
written by byunggook.na (SAIT)
reference : nequip.model._build.py
"""

import torch
import inspect

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.datasets import LmdbDataset 

from nequip.utils.config import Config
from nequip.data import AtomicDataDict, AtomicData
from nequip.data.transforms import TypeMapper

# pre-defined modules in NequIP
from nequip.model import SimpleIrrepsConfig, ForceOutput, PartialForceOutput

# modified modules to enable to be compatible with LMDB datasets
from src.models.nequip.energy_model import EnergyModel
from src.models.nequip.rescale import RescaleEnergyEtc, PerSpeciesRescale
from src.datasets.nequip.statistics import (
    compute_avg_num_neighbors, 
    compute_global_shift_and_scale,
    compute_per_species_shift_and_scale
)
from src.common.utils import bm_logging # benchmark logging


def compute_statistics(model_config, type_mapper, dataset_name, global_rescale_shift=None):
    # add statistics results to config
    dataset = LmdbDataset(dataset_name)

    # 1) avg_num_neighbors (required by EnergyModel)
    model_config["avg_num_neighbors"] = compute_avg_num_neighbors(
        config=model_config, 
        initialize=True, 
        dataset=dataset,
        transform=type_mapper,
    )

    # 2) per_species_rescale_shifts and per_species_rescale_scales (required by PerSpeciesRescale)
    if "PerSpeciesRescale" in model_config["model_builders"]:
        shifts, scales, arguments_in_dataset_units = compute_per_species_shift_and_scale(
            config=model_config, 
            initialize=True, 
            dataset=dataset,
            transform=type_mapper,
        )
        model_config["per_species_rescale_shifts"] = shifts
        model_config["per_species_rescale_scales"] = scales
        model_config["arguments_in_dataset_units"] = arguments_in_dataset_units

    # 3) global_rescale_shift and global_rescale_scale (required by RescaleEnergyEtc (i.e., GlobalRescale))
    if ("RescaleEnergyEtc" in model_config["model_builders"] or
        "GlobalRescale" in model_config["model_builders"]
    ):
        if global_rescale_shift is not None:
            default_shift_keys = [AtomicDataDict.TOTAL_ENERGY_KEY]
            bm_logging.warning(
                f"!!!! Careful global_shift is set to {global_rescale_shift}."
                f"The model for {default_shift_keys} will no longer be size extensive"
            )
        global_shift, global_scale = compute_global_shift_and_scale(
            config=model_config, 
            initialize=True, 
            dataset=dataset,
            transform=type_mapper,
        )
        model_config["global_rescale_shift"] = global_shift
        model_config["global_rescale_scale"] = global_scale

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
        cutoff=5.0,
        max_neighbors=None, # not used?
        use_pbc=True,
        regress_forces=True,
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
        avg_num_neighbors="auto",
        per_species_rescale_shifts_trainable=False,
        per_species_rescale_scales_trainable=False,
        per_species_rescale_shifts="dataset_per_atom_total_energy_mean",
        per_species_rescale_scales="dataset_forces_rms",
        global_rescale_shift_trainable=False,
        global_rescale_scale_trainable=False,
        global_rescale_shift=None,
        global_rescale_scale="dataset_forces_rms",
    ):
        self.num_targets = num_targets
        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
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
        model_config = compute_statistics(
            model_config = model_config, 
            type_mapper = self.type_mapper, 
            dataset_name = dataset, 
            global_rescale_shift=global_rescale_shift,
        )

        # constrcut the NequIP model
        builders = [eval(module) for module in model_config["model_builders"]]
        self.nequip_model = initiate_model_by_builders(
            builders=builders, 
            config=model_config, 
            initialize=True,
        )

        # maintain rescale layers individually
        self.rescale_layers = []
        outer_layer = self.nequip_model
        while hasattr(outer_layer, "unscale"):
            self.rescale_layers.append(outer_layer)
            outer_layer = getattr(outer_layer, "model", None)

    def do_unscale(self, data, force_process=False):
        # rescaling
        # do copy in unscale()
        # assert AtomicDataDict.TOTAL_ENERGY_KEY in data.keys() and AtomicDataDict.FORCE_KEY in data.keys()
        for layer in self.rescale_layers:
            data = layer.unscale(data, force_process=force_process)
        return data

    def undo_unscale(self, data, force_process=False):
        # undo rescaling
        # do copy in scale()
        # assert AtomicDataDict.TOTAL_ENERGY_KEY in out.keys() and AtomicDataDict.FORCE_KEY in out.keys()
        for layer in self.rescale_layers[::-1]:
            data = layer.scale(data, force_process=force_process)
        return data

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # data is already moved to device 
        # by OCPDataParallel (ocpmodels/common/data_parallel.py)
        #data = self.type_mapper.transform(data.to_dict()) 
        input_data = AtomicData.to_AtomicDataDict(data)

        # This is not necessary in model forward,
        # because this changes energy/force/atomic_energy (i.e., ground target).
        #input_data = self.do_unscale(data)

        # model forward
        # : output is dict
        out = self.nequip_model(input_data)
        
        # return values required in an OCP-based trainer
        if self.regress_forces:
            return out[AtomicDataDict.TOTAL_ENERGY_KEY], out[AtomicDataDict.FORCE_KEY]
        else:
            return out[AtomicDataDict.TOTAL_ENERGY_KEY]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.nequip_model.parameters())
