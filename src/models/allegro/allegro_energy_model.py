"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.nn.radial_basis import BesselBasis
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    Allegro_Module,
    ScalarMLP,
)
from allegro._keys import EDGE_FEATURES, EDGE_ENERGY


# reference : Allegro() in allegro/allegro/model/_allegro.py
# We modified the function to enable to be compatible with LMDB datasets
# : avg_num_neighbors is added to the config before constructing AllegroEnergyModel
def AllegroEnergyModel(config):
    # Handle simple irreps
    assert "l_max" in config
    l_max = int(config["l_max"])
    parity_setting = config["parity"]
    assert parity_setting in ("o3_full", "o3_restricted", "so3")
    irreps_edge_sh = repr(
        o3.Irreps.spherical_harmonics(
            l_max, p=(1 if parity_setting == "so3" else -1)
        )
    )
    nonscalars_include_parity = parity_setting == "o3_full"
    # check consistant
    assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
    assert (
        config.get("nonscalars_include_parity", nonscalars_include_parity)
        == nonscalars_include_parity
    )
    config["irreps_edge_sh"] = irreps_edge_sh
    config["nonscalars_include_parity"] = nonscalars_include_parity

    layers = {
        # -- Encode --
        # Get various edge invariants
        "one_hot": OneHotAtomEncoding,
        "radial_basis": (
            RadialBasisEdgeEncoding,
            dict(
                basis=(
                    NormalizedBasis
                    if config.get("normalize_basis", True)
                    else BesselBasis
                ),
                out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            ),
        ),
        # Get edge nonscalars
        "spharm": SphericalHarmonicEdgeAttrs,
        # The core allegro model:
        "allegro": (
            Allegro_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            ),
        ),
        "edge_eng": (
            ScalarMLP,
            dict(field=EDGE_FEATURES, out_field=EDGE_ENERGY, mlp_output_dimension=1),
        ),
        # Sum edgewise energies -> per-atom energies:
        "edge_eng_sum": EdgewiseEnergySum,
        # Sum system energy:
        "total_energy_sum": (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        ),
    }

    return SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)
