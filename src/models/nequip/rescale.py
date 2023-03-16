
from typing import List, Optional, Union

import torch

from nequip.nn import RescaleOutput, GraphModuleMixin, PerSpeciesScaleShift
from nequip.data import AtomicDataDict
from nequip.data.transforms import TypeMapper

def RescaleEnergyEtc(
    model: GraphModuleMixin,
    config,
):
    return GlobalRescale(
        model=model,
        config=config,
        module_prefix="global_rescale",
        default_scale_keys=AtomicDataDict.ALL_ENERGY_KEYS,
        default_shift_keys=[AtomicDataDict.TOTAL_ENERGY_KEY],
        default_related_scale_keys=[AtomicDataDict.PER_ATOM_ENERGY_KEY],
        default_related_shift_keys=[],
    )


def GlobalRescale(
    model: GraphModuleMixin,
    config,
    module_prefix: str,
    default_scale_keys: list,
    default_shift_keys: list,
    default_related_scale_keys: list,
    default_related_shift_keys: list,
):
    """Add global rescaling for energy(-based quantities).
    
    Statistics are computed in the constructor of NequIPWrap class (src.models.nequip.nequip.py)
    """
    return RescaleOutput(
        model=model,
        scale_keys=[k for k in default_scale_keys if k in model.irreps_out],
        scale_by=config[f"{module_prefix}_scale"],
        shift_keys=[k for k in default_shift_keys if k in model.irreps_out],
        shift_by=config[f"{module_prefix}_shift"],
        related_scale_keys=default_related_scale_keys,
        related_shift_keys=default_related_shift_keys,
        shift_trainable=config.get(f"{module_prefix}_shift_trainable", False),
        scale_trainable=config.get(f"{module_prefix}_scale_trainable", False),
    )


def PerSpeciesRescale(
    model: GraphModuleMixin,
    config,
):
    """Add global rescaling for energy(-based quantities).

    Statistics are computed in the constructor of NequIPWrap class (src.models.nequip.nequip.py)
    """
    model.insert_from_parameters(
        before="total_energy_sum",
        name="per_species_rescale",
        shared_params=config,
        builder=PerSpeciesScaleShift,
        params=dict(
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            shifts=config["per_species_rescale_shifts"],
            scales=config["per_species_rescale_scales"],
            arguments_in_dataset_units=config["arguments_in_dataset_units"],
        ),
    )
    return model