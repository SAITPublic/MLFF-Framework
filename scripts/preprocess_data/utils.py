"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os
import json
import numpy as np


def parse_file_path(file_path):
    filename = file_path.split("/")[-1]
    f_extension = filename.split(".")[-1]
    assert f_extension in ["xyz", "extxyz"], "The file extention should be .xyz or .extxyz"
    f_name = filename.replace(f".{f_extension}", "")
    return f_name, f_extension


def generate_lmdb_path(out_path, out_f_name, r_max, max_neighbors):
    if r_max is None:
        path = os.path.join(out_path, f"atom_cloud")
    else:
        if max_neighbors is None:
            path = os.path.join(out_path, f"atom_graph_rmax{r_max}")
        else:
            path = os.path.join(out_path, f"atom_graph_rmax{r_max}_maxneighbor{max_neighbors}")
    return os.path.join(path, f"{out_f_name}.lmdb")


def save_normalization_statistics(trajectory, out_dir, energy_type="free_energy"):
    # extract meta data which includes normalization statistics
    # we can use free energy for our dataset (according to the comment of Prof. Han)
    assert energy_type in ["total_energy", "free_energy"]
    if energy_type == "total_energy":
        energy_type = "energy"
    energies = np.array([atoms.info[energy_type] for atoms in trajectory]) 
    energies_per_atom = np.array([atoms.info[energy_type]/atoms.get_forces().shape[0] for atoms in trajectory]) 
    forces = np.concatenate([atoms.get_forces() for atoms in trajectory])
    norm_stats = {
        "energy_mean": energies.mean(),
        "energy_std" : energies.std(),
        "energy_per_atom_mean":energies_per_atom.mean(),
        "energy_per_atom_std" : energies_per_atom.std(),
        "force_mean" : 0.0,
        "force_std"  : forces.std(),
    }
    with open(os.path.join(out_dir, "normalize_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=4)
