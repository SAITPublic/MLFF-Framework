"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import torch

from nequip.utils.torch_geometric.batch import Batch as BatchNequIP
from nequip.data import AtomicData, AtomicDataDict

from src.common.utils import bm_logging
from src.common.collaters.parallel_collater import ParallelCollater


def convert_ocp_Data_into_nequip_AtomicData(ocp_data, transform):
    kwargs = {}
    
    # atomic_types ???

    # atomic_numbers
    # : (nAtoms) -> (nAtoms, 1)
    kwargs["atomic_numbers"] = ocp_data.atomic_numbers.unsqueeze(-1)

    # cell
    # : (1, 3, 3) -> (3, 3)
    kwargs["cell"] = ocp_data.cell.view(3, 3)

    # edge_cell_shift
    kwargs["edge_cell_shift"] = ocp_data.cell_offsets.type(ocp_data.pos.dtype)

    # edge_index
    # : change the order of src and dest which is different from that of OCP Data
    kwargs["edge_index"] = ocp_data.edge_index[[1, 0]] 
    
    # forces
    if hasattr(ocp_data, 'force') and ocp_data.y is not None:
        kwargs["forces"] = ocp_data.force

    # total_energy and free_energy (which are identical for now)
    if hasattr(ocp_data, 'y') and ocp_data.y is not None:
        kwargs["total_energy"] = torch.tensor(ocp_data.y)
        kwargs["free_energy"] = torch.tensor(ocp_data.y)

    # pos
    kwargs["pos"] = ocp_data.pos

    # initiate AtomicData
    data = AtomicData(**kwargs)

    # lastly, apply the transfrom defined in the type mapper
    # : atomic_numbers -> atom_types
    data = transform(data)

    # Additional information used in an OCP-based trainer
    # fixed atoms
    if hasattr(ocp_data, 'fixed') and ocp_data.fixed is not None:
        data.fixed = ocp_data.fixed

    return data


class ParallelCollaterNequIP(ParallelCollater):
    def __init__(self, num_gpus, otf_graph=False, transform=None, use_pbc=False, type_mapper=None):
        self.num_gpus = num_gpus
        self.otf_graph = otf_graph
        self.transform = transform # TypeMapper in NequIP
        self.use_pbc = use_pbc # torch.tensor([use_pbc] * 3)
        self.type_mapper = type_mapper

    def data_list_collater(self, data_list, otf_graph=False):
        # First, generate a list of AtomicData.
        # : from data_list (= a list of pytorch_geometric Data loaded from LMDB)
        atomic_data_list = [
            convert_ocp_Data_into_nequip_AtomicData(ocp_data=d, transform=self.type_mapper)
            for d in data_list
        ]
        # Then, the list of AtomicData will be a batch by using 'from_data_list' defined in NequIP
        # : from_data_list customized by NequIP (nequip/utils/torch_geometric/batch.py)
        batch = BatchNequIP.from_data_list(atomic_data_list)

        # convert dimension of energy in a batch to be compatible with an OCP trainer
        # (batch size, 1) -> (batch_size)
        # batch.free_energy is the shallow copy of batch.total_energy
        batch.total_energy = batch.total_energy.view(-1)

        if not otf_graph:
            batch = self.set_neighbors_in_a_batch(data_list, batch)
        return batch