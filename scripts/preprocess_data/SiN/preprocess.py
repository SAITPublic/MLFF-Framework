"""
Modified by byunggook.na (reference: scripts/preprocess_ef.py)

Creates LMDB files with extracted graph features from provided *.extxyz files of SiN
for the S2EF task.
"""

import sys
sys.path.insert(0, "../../../codebases/ocp/")

import os
import argparse
import random
import pickle
import json
import lmdb
import csv
from tqdm import tqdm

import numpy as np
import torch

import ase
from ase import io

from ocpmodels.preprocessing import AtomsToGraphs


def write_extxyz_to_lmdb(args):
    # for split case 1
    split_case = 1

    split_names = {'Trainset': 'train', 'Validset': 'valid', 'Testset': 'test'}
    for split_name in split_names.keys():
        xyzfile = os.path.join(args.data_dir, f"{split_name}_{split_case}.xyz")
        assert os.path.isfile(xyzfile), f"{npzfile} does not exist."

    out_path = os.path.join(args.out_path, f"split_{split_case}")
    if args.r_max is None:
        out_path = os.path.join(out_path, f"atom_cloud")
        r_edges = False
    else:
        r_edges = True
        if args.max_neighbors is None:
            out_path = os.path.join(out_path, f"atom_graph_rmax{args.r_max}")
        else:
            out_path = os.path.join(out_path, f"atom_graph_rmax{args.r_max}_maxneighbor{args.max_neighbors}")
    os.makedirs(out_path, exist_ok=True)

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=100000000 if args.max_neighbors is None else args.max_neighbors, 
        radius=args.r_max,
        r_energy=True, 
        r_forces=True, 
        r_fixed=True,
        r_distances=False,
        r_edges=r_edges, 
    )

    print(f"Start preprocessing")

    for split_name, split_name_val in split_names.items():
        # load xyz as ase.Atoms
        xyzfile = os.path.join(args.data_dir, f"{split_name}_{split_case}.xyz")
        trajectory = ase.io.read(xyzfile, index=":", format="extxyz")

        if split_name == "Trainset":
            # extract meta data which includes normalization statistics
            energies = np.array([atoms.get_potential_energy() for atoms in trajectory])
            forces = np.concatenate([atoms.get_forces() for atoms in trajectory])
            norm_stats = {
                "energy_mean": energies.mean(),
                "energy_std" : energies.std(),
                "force_mean" : 0.0,
                "force_std"  : forces.std(),
            }
            with open(os.path.join(out_path, "normalize_stats.json"), "w", encoding="utf-8") as f:
                json.dump(norm_stats, f, ensure_ascii=False, indent=4)
        
        # save lmdb
        lmdbfile = os.path.join(out_path, f"{split_name_val}.lmdb")
        db = lmdb.open(
            lmdbfile,
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
            lock=False,
        )   

        for i, snapshot in enumerate(tqdm(trajectory)):
            data = a2g.convert(snapshot)
            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
        
        db.sync()
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        help="Path of directory that includes *.xyz of SiN",
    )
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create at --data-dir if doesn't exist",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="If the cutoff radius is set, output LMDB files include edge index information, which means otf_graph in config files can be False.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=None,
        help="The maximum number of neighbors",
    )
    args = parser.parse_args()

    write_extxyz_to_lmdb(args)


if __name__ == "__main__":
    main()