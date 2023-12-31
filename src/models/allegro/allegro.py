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
from nequip.model import ForceOutput, PartialForceOutput, StressForceOutput

from src.common.utils import bm_logging
from src.models.nequip.rescale import RescaleEnergyEtc, PerSpeciesRescale
from src.models.nequip.nequip import (
    set_model_config_based_on_data_statistics, 
    initiate_model_by_builders
)
from src.models.nequip.utils import (
    compute_avg_num_neighbors, 
    compute_global_shift_and_scale,
    compute_per_species_shift_and_scale
)
from src.models.allegro.allegro_energy_model import AllegroEnergyModel as Allegro


@registry.register_model("allegro")
class AllegroWrap(BaseModel):
    def __init__(
        self,
        num_atoms, # not used
        bond_feat_dim, # not used
        num_targets,
        cutoff=6.0, # r_max
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
        avg_num_neighbors="auto",
        # architecture arguments
        model_builders=[
            "Allegro",
            "PerSpeciesRescale",
            "ForceOutput",
            "RescaleEnergyEtc",
        ],
        num_layers=3,
        l_max=1,
        parity="o3_full", 
        BesselBasis_trainable=True,
        PolynomialCutoff_p=6,
        env_embed_multiplicity=32, # num features
        env_embed_mlp_latent_dimensions=[],
        env_embed_mlp_nonlinearity=None,
        env_embed_mlp_initialization="uniform",
        embed_initial_edge=True,
        two_body_latent_mlp_latent_dimensions=[64, 128, 256, 512],
        two_body_latent_mlp_nonlinearity="silu",
        two_body_latent_mlp_initialization="uniform",
        latent_mlp_latent_dimensions=[512],
        latent_mlp_nonlinearity="silu",
        latent_mlp_initialization="uniform",
        latent_resnet=True,
        edge_eng_mlp_latent_dimensions=[128],
        edge_eng_mlp_nonlinearity=None,
        edge_eng_mlp_initialization="uniform",
        # NequIP default arguments used in Allegro
        num_basis=8, # BesselBasis
        per_species_rescale_shifts_trainable=False, # PerSpeciesRescale
        per_species_rescale_scales_trainable=False, # PerSpeciesRescale
        per_species_rescale_shifts="dataset_per_atom_total_energy_mean", # PerSpeciesRescale
        per_species_rescale_scales="dataset_forces_rms", # PerSpeciesRescale
        global_rescale_shift_trainable=False, # RescaleEnergyEtc
        global_rescale_scale_trainable=False, # RescaleEnergyEtc
        global_rescale_shift=None, # RescaleEnergyEtc
        global_rescale_scale="dataset_forces_rms", # RescaleEnergyEtc
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
            raise NotImplementedError("on-the-fly garph generation is not enabled for Allegro")
        
        self.max_neighbors = max_neighbors
        super().__init__()

        model_config = dict(
            model_builders=model_builders,
            r_max=self.cutoff,
            num_layers=num_layers,
            l_max=l_max,
            parity=parity,
            BesselBasis_trainable=BesselBasis_trainable,
            PolynomialCutoff_p=PolynomialCutoff_p,
            env_embed_multiplicity=env_embed_multiplicity, # num features
            env_embed_mlp_latent_dimensions=env_embed_mlp_latent_dimensions,
            env_embed_mlp_nonlinearity=env_embed_mlp_nonlinearity,
            env_embed_mlp_initialization=env_embed_mlp_initialization,
            embed_initial_edge=embed_initial_edge,
            two_body_latent_mlp_latent_dimensions=two_body_latent_mlp_latent_dimensions,
            two_body_latent_mlp_nonlinearity=two_body_latent_mlp_nonlinearity,
            two_body_latent_mlp_initialization=two_body_latent_mlp_initialization,
            latent_mlp_latent_dimensions=latent_mlp_latent_dimensions,
            latent_mlp_nonlinearity=latent_mlp_nonlinearity,
            latent_mlp_initialization=latent_mlp_initialization,
            latent_resnet=latent_resnet,
            edge_eng_mlp_latent_dimensions=edge_eng_mlp_latent_dimensions,
            edge_eng_mlp_nonlinearity=edge_eng_mlp_nonlinearity,
            edge_eng_mlp_initialization=edge_eng_mlp_initialization,
            avg_num_neighbors=avg_num_neighbors,
            ## belows are default
            num_basis=num_basis,
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
        self.allegro_model = initiate_model_by_builders(
            builders=builders, 
            config=model_config, 
            initialize=initialize,
        )

        # maintain rescale layers individually
        self.rescale_layers = []
        outer_layer = self.allegro_model
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
        out = self.allegro_model(input_data)
        
        # return values required in an OCP-based trainer
        if self.regress_stress:
            return out[AtomicDataDict.TOTAL_ENERGY_KEY], out[AtomicDataDict.FORCE_KEY], out[AtomicDataDict.STRESS_KEY]
        elif self.regress_forces:
            return out[AtomicDataDict.TOTAL_ENERGY_KEY], out[AtomicDataDict.FORCE_KEY]
        else:
            return out[AtomicDataDict.TOTAL_ENERGY_KEY]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.allegro_model.parameters())
