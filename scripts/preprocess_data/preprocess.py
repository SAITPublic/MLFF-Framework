"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(1, os.path.abspath("../../codebases/ocp/"))

import pickle
import lmdb
from tqdm import tqdm
from pathlib import Path

import ase
from ase import io

from src.preprocessing.atoms_to_graphs import AtomsToGraphsWithTolerance

from data_flags import DataFlags
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
        ), "If --data is empty, at least one of three data split (--train-data, --valid-data, --test-data) should be given."
    else:
        assert (args.train_data is None
                and args.valid_data is None
                and args.test_data is None
        ), "If --data is given, all of --train-data, --valid-data, and --test-data should be empty."

    if args.data is None:
        res = []
        for f_name, data in zip(
            [args.train_data_output_name, args.valid_data_output_name, args.test_data_output_name],
            [args.train_data, args.valid_data, args.test_data]
        ):
            if data is not None:
                _, f_extension = parse_file_path(file_path=data)
                out_lmdb_path = generate_lmdb_path(
                    out_path=args.out_path,
                    out_f_name=f_name,
                    r_max=args.r_max,
                    max_neighbors=args.max_neighbors,
                )
                res.append((f_name, data, out_lmdb_path))
        return res
    else:
        f_name, f_extension = parse_file_path(file_path=args.data)
        if args.data_output_name is not None:
            f_name = args.data_output_name
        out_lmdb_path = generate_lmdb_path(
            out_path=args.out_path,
            out_f_name=f_name,
            r_max=args.r_max,
            max_neighbors=args.max_neighbors,
        )
        return [(f_name, args.data, out_lmdb_path)]
        
# reference: ocp/scripts/preprocess_ef.py
def write_extxyz_to_lmdb(args):
    # prepare input xyz and output lmdb pairs
    key_xyz_lmdb_list = prepare_key_xyz_lmdb_list(args)

    # make a directory where output lmdb is saved
    out_dir = Path(key_xyz_lmdb_list[0][2]).parent
    os.makedirs(out_dir, exist_ok=True)

    # Initialize feature extractor.
    a2g = AtomsToGraphsWithTolerance(
        max_neigh=2000000000 if args.max_neighbors is None else args.max_neighbors, 
        radius=args.r_max,
        r_energy=True, 
        r_forces=True,
        r_stress=args.get_stress,
        r_fixed=True,
        r_distances=False,
        r_edges=(args.r_max is not None), # if r_max is given, graph data with edges is generated
        tolerance=1e-8,
    )

    # Start preprocessing
    for key, xyzfile, lmdbfile in key_xyz_lmdb_list:
        trajectory = ase.io.read(xyzfile, index=":", format="extxyz")
        if key == "train" and args.save_normalization:
            save_normalization_statistics(
                trajectory=trajectory, 
                out_dir=out_dir,
                energy_type=args.energy_type,
            )
        
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
        for i, snapshot in enumerate(tqdm(trajectory)):
            data = a2g.convert(snapshot)
            # explictly save both total_energy and free_energy that exist in the data
            if "energy" in snapshot.info:
                data.total_energy = snapshot.info["energy"] # total energy
            if "free_energy" in snapshot.info:
                data.free_energy = snapshot.info["free_energy"] # free energy
            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()
        
        # close lmdb
        db.sync()
        db.close()


if __name__ == "__main__":
    flags = DataFlags()
    args = flags.parser.parse_args()

    write_extxyz_to_lmdb(args)
