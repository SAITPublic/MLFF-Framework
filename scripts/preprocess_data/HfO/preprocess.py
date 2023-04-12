"""
Written by byunggook.na 
(reference: scripts/preprocess_ef.py)

Creates LMDB files with extracted graph features from provided *.extxyz files
for the S2EF task.
"""

import sys
sys.path.insert(0, "../../../codebases/ocp/")

import os
import argparse
import pickle
import lmdb
import csv
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch

import ase
from ase import io

from ocpmodels.preprocessing import AtomsToGraphs

sys.path.insert(1, "../")
from utils import (
    parse_file_path, 
    generate_lmdb_path,
    save_normalization_statistics,
)


def prepare_key_xyz_lmdb_list(args):
    if args.data is None:
        assert (args.train_data is not None
                or args.valid_data is not None 
                or args.test_data is not None
        ), "If --data is empty, --train-data or --valid-data or --test-data should be given."
    else:
        assert (args.train_data is None
                and args.valid_data is None
                and args.test_data is None
        ), "If --data is given, all of --train-data, --valid-data, and --test-data should be empty."

    if args.data is None:
        res = []
        for key, data in zip(
            ['train', 'valid', 'test'],
            [args.train_data, args.valid_data, args.test_data]
        ):
            if data is not None:
                f_name, f_extension = parse_file_path(file_path=data)
                out_lmdb_path = generate_lmdb_path(
                    out_path=args.out_path,
                    out_f_name=key,
                    r_max=args.r_max,
                    max_neighbors=args.max_neighbors,
                )
                res.append((key, data, out_lmdb_path))
        return res
    else:
        f_name, f_extension = parse_file_path(file_path=args.data)
        out_lmdb_path = generate_lmdb_path(
            out_path=args.out_path,
            out_f_name=f_name,
            r_max=args.r_max,
            max_neighbors=args.max_neighbors,
        )
        return [(f_name, args.data, out_lmdb_path)]
        

def write_extxyz_to_lmdb(args):
    # prepare input xyz and output lmdb pairs
    key_xyz_lmdb_list = prepare_key_xyz_lmdb_list(args)

    # make a directory where output lmdb is saved
    out_dir = Path(key_xyz_lmdb_list[0][2]).parent
    os.makedirs(out_dir, exist_ok=True)

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=100000000 if args.max_neighbors is None else args.max_neighbors, 
        radius=args.r_max,
        r_energy=True, 
        r_forces=True, 
        r_fixed=True,
        r_distances=False,
        r_edges=(args.r_max is not None), # if r_max is given, graph data with edges is generated
    )

    # Start preprocessing
    for key, xyzfile, lmdbfile in key_xyz_lmdb_list:
        trajectory = ase.io.read(xyzfile, index=":", format="extxyz")
        if key == "train":
            save_normalization_statistics(trajectory, out_dir)
        
        # open lmdb 
        db = lmdb.open(
            lmdbfile,
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
            lock=False,
        )

        # preprocess and push data into lmdb
        print(f"Start preprocess {xyzfile}")
        print(f"Its lmdb is saved at {lmdbfile}")
        for i, snapshot in enumerate(tqdm(trajectory)):
            data = a2g.convert(snapshot)
            # explictly save two types of energy, called total_energy and free_energy.
            data.total_energy = snapshot.info["energy"]
            data.free_energy = snapshot.info["free_energy"]
            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
        
        # close lmdb
        db.sync()
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path of a single file (.xyz or .extxyz)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path of a train file (.xyz or .extxyz)",
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        default=None,
        help="Path of a valid file (.xyz or .extxyz)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path of a test file (.xyz or .extxyz)",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Directory to save output data. If not given, the output data is saved at a parent directory where --data or --train-data exist",
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
