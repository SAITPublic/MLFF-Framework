"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import torch
from torch_geometric.data import Batch

from mace.data import AtomicData
from mace.tools import atomic_numbers_to_indices, to_one_hot
from mace.tools.torch_geometric.batch import Batch as BatchMACE

from src.common.utils import bm_logging
from src.common.collaters.parallel_collater import ParallelCollater


def convert_ocp_Data_into_mace_AtomicData(ocp_data, z_table):
    kwargs = {}

    # edge_index
    # : change the order of src and dest which is different from that of OCP Data
    # : MACE order = NequIP order
    kwargs["edge_index"] = ocp_data.edge_index[[1, 0]] 

    # node_attrs
    # : one-hot encoding for initial node attributes
    atomic_numbers = ocp_data.atomic_numbers # (nAtoms)
    indices = atomic_numbers_to_indices(atomic_numbers, z_table=z_table)
    one_hot = to_one_hot(
        torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )
    kwargs["node_attrs"] = one_hot

    # pos
    kwargs["positions"] = ocp_data.pos
    
    # cell
    # : (1, 3, 3) -> (3, 3)
    kwargs["cell"] = ocp_data.cell.view(3, 3)

    # unit_shifts
    kwargs["unit_shifts"] = ocp_data.cell_offsets.type(ocp_data.pos.dtype)

    # shift
    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    kwargs["shifts"] = torch.mm(kwargs["unit_shifts"], kwargs["cell"])  # [n_edges, 3]
    
    # forces
    kwargs["forces"] = ocp_data.force if hasattr(ocp_data, 'force') and ocp_data.y is not None else None

    # total_energy and free_energy (which are identical for now)
    kwargs["energy"] = torch.tensor(ocp_data.y) if hasattr(ocp_data, 'y') and ocp_data.y is not None else None
    # kwargs["free_energy"] = torch.tensor(ocp_data.y)

    # the arguments below are not used in this benchmark
    kwargs['weight'] = None
    kwargs['energy_weight'] = None
    kwargs['forces_weight'] = None
    kwargs['stress_weight'] = None
    kwargs['virials_weight'] = None
    kwargs['stress'] = None
    kwargs['virials'] = None
    kwargs['dipole'] = None
    kwargs['charges'] = None

    # initiate AtomicData
    data = AtomicData(**kwargs)

    # for free_energy
    data.free_energy = torch.tensor(ocp_data.y) if hasattr(ocp_data, 'y') and ocp_data.y is not None else None

    # Additional information used in an OCP-based trainer
    # fixed atoms
    if hasattr(ocp_data, 'fixed') and ocp_data.fixed is not None:
        data.fixed = ocp_data.fixed

    return data


class ParallelCollaterMACE(ParallelCollater):
    def __init__(self, num_gpus, otf_graph=False, use_pbc=False, z_table=None):
        self.num_gpus = num_gpus
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc # torch.tensor([use_pbc] * 3)
        assert z_table is not None
        self.z_table = z_table

    def data_list_collater(self, data_list, otf_graph=False):
        # First, generate a list of Configuration (which is data structure used in MACE).
        # : from data_list (= a list of pytorch_geometric Data loaded from LMDB)
        atomic_data_list = [
            convert_ocp_Data_into_mace_AtomicData(ocp_data=d, z_table=self.z_table)
            for d in data_list
        ]
        # Then, the list of AtomicData will be a batch by using 'from_data_list' defined in MACE
        # : from_data_list customized by MACE (mace/tools/torch_geometric/batch.py)
        batch = BatchMACE.from_data_list(atomic_data_list)
        # batch.energy # shape: (batch_size)

        if not otf_graph:
            batch = self.set_neighbors_in_a_batch(data_list, batch)
        return batch