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

from src.common.utils import bm_logging


# reference : ocp/ocpmodels/common/data_parallel.py
class ParallelCollater:
    def __init__(self, num_gpus, otf_graph=False):
        self.num_gpus = num_gpus
        self.otf_graph = otf_graph

    def set_neighbors_in_a_batch(self, data_list, batch):
        try:
            # edge_index shape : (2, numEdges)
            n_neighbors = [data.edge_index.shape[1] for data in data_list]
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            bm_logging.warning("LMDB does not contain edge index information, set otf_graph=True")
        return batch

    def data_list_collater(self, data_list, otf_graph=False):
        batch = Batch.from_data_list(data_list)
        if batch.y.dtype == torch.float64:
            batch.y = batch.y.float() # float64 -> float32
        if not otf_graph:
            batch = self.set_neighbors_in_a_batch(data_list, batch)
        return batch

    def __call__(self, data_list):
        if self.num_gpus in [0, 1]:  # adds cpu-only case
            batch = self.data_list_collater(data_list, otf_graph=self.otf_graph)
            return [batch]

        else:
            num_devices = min(self.num_gpus, len(data_list))

            count = torch.tensor([data.num_nodes for data in data_list])
            cumsum = count.cumsum(0)
            cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
            device_id = num_devices * cumsum.to(torch.float) / cumsum[-1].item()
            device_id = (device_id[:-1] + device_id[1:]) / 2.0
            device_id = device_id.to(torch.long)
            split = device_id.bincount().cumsum(0)
            split = torch.cat([split.new_zeros(1), split], dim=0)
            split = torch.unique(split, sorted=True)
            split = split.tolist()

            return [
                self.data_list_collater(data_list[split[i] : split[i + 1]])
                for i in range(len(split) - 1)
            ]
