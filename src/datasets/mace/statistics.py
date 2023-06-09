"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import numpy as np
import torch

from mace.tools import to_numpy
from mace.tools.scatter import scatter_sum
from mace.modules.blocks import AtomicEnergiesBlock


# reference : compute_average_E0s() in mace/mace/data/utils.py
def compute_average_E0s(train_dataset, z_table):
    len_train = len(train_dataset)
    len_zs = len(z_table)
    A = np.zeros((len_train, len_zs))
    B = np.zeros(len_train)
    for i in range(len_train):
        B[i] = train_dataset[i].y
        for j, z in enumerate(z_table.zs):
            A[i, j] = np.count_nonzero(train_dataset[i].atomic_numbers == z)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = E0s[i]
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute E0s using least squares regression, using the same for all atoms"
        )
        atomic_energies_dict = {}
        for i, z in enumerate(z_table.zs):
            atomic_energies_dict[z] = 0.0
    return atomic_energies_dict


# reference : compute_avg_num_neighbors() in mace/mace/modules/utils.py
def compute_avg_num_neighbors(data_loader):
    num_neighbors = []
    for batch_list in data_loader:
        for batch in batch_list:
            _, receivers = batch.edge_index
            _, counts = torch.unique(receivers, return_counts=True)
            num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


# reference : compute_mean_std_atomic_inter_energy() in mace/mace/modules/utils.py
def compute_mean_std_atomic_inter_energy(data_loader, atomic_energies):
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch_list in data_loader:
        for batch in batch_list:
            node_e0 = atomic_energies_fn(batch.node_attrs)
            graph_e0s = scatter_sum(
                src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
            )
            graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
            avg_atom_inter_es_list.append(
                (batch.energy - graph_e0s) / graph_sizes
            )  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()

    return mean, std


# reference : compute_mean_rms_energy_forces() in mace/mace/modules/utils.py
def compute_mean_rms_energy_forces(data_loader, atomic_energies):
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch_list in data_loader:
        for batch in batch_list:
            node_e0 = atomic_energies_fn(batch.node_attrs)
            graph_e0s = scatter_sum(
                src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
            )
            graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
            atom_energy_list.append(
                (batch.energy - graph_e0s) / graph_sizes
            )  # {[n_graphs], }
            forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

    return mean, rms