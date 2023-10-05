"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor

from ocpmodels.preprocessing.atoms_to_graphs import AtomsToGraphs

import ase.db.sqlite
import ase.io.trajectory
import numpy as np
import torch
from torch_geometric.data import Data
from ocpmodels.common.utils import collate
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress

class AtomsToGraphsWithTolerance(AtomsToGraphs):
    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_distances=False,
        r_edges=True,
        r_fixed=True,
        r_pbc=False,
        tolerance=1e-8,
    ):
        super().__init__(
            max_neigh=max_neigh,
            radius=radius,
            r_energy=r_energy,
            r_forces=r_forces,
            r_distances=r_distances,
            r_edges=r_edges,
            r_fixed=r_fixed,
            r_pbc=r_pbc,
        )
        
        self.r_stress=r_stress
        # set the numerical tolerance which will be used to exclude self-edge when obtaining neighbors
        self.tolerance = tolerance

    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=self.tolerance, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets
    def convert(self,atoms,):
        data=super().convert(atoms)
        if self.r_stress:
            stress = torch.Tensor(voigt_6_to_full_3x3_stress(atoms.get_stress(apply_constraint=False))).unsqueeze(0)
            data.stress = stress
        return data