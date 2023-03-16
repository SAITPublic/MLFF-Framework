"""
Modified by byunggook.na (reference: ocp/scripts/preprocess_ef.py)

Creates LMDB files with extracted graph features from provided *.npz files of rMD17
for the S2EF task.
"""

import sys
sys.path.insert(0, "/nas/ocp")

import os
import argparse
import random
import pickle
import json
import lmdb
from tqdm import tqdm

import numpy as np
import torch

import ase
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from ocpmodels.preprocessing import AtomsToGraphs

molecules = [
    "aspirin", 
    "azobenzene", 
    "benzene", 
    "ethanol", 
    "malonaldehyde", 
    "naphthalene", 
    "paracetamol", 
    "salicylic", 
    "toluene", 
    "uracil"
    ]

kcalPerMol_FOR_1eV = 23.060548 ## ref: http://wild.life.nctu.edu.tw/class/common/energy-unit-conv-table-detail.html


def convert_symbols(nuclear_charges):
    nuclear_charges = nuclear_charges.tolist()
    nuclear_charges.append("END")
    charge_dict = {1:"H", 6:"C", 7:"N", 8:"O"}
    symbols_list = []
    symbols = ''
    prev_symbol = None
    prev_symbol_count = 0
    for c in nuclear_charges:
        if c == "END":
            symbols += f"{prev_symbol}{prev_symbol_count}"
            return symbols, symbols_list
            
        curr_symbol = charge_dict[c]
        symbols_list.append(curr_symbol)
        if curr_symbol == prev_symbol:
            prev_symbol_count += 1
        else:
            if prev_symbol is not None:
                symbols += f"{prev_symbol}{prev_symbol_count}"
            prev_symbol = curr_symbol
            prev_symbol_count = 1


def write_npz_to_lmdb(args, mol):

    npzfile = os.path.join(args.data_dir, f"rmd17_{mol}.npz")
    assert os.path.isfile(npzfile), f"{npzfile} does not exist. Please download rMD17 according to README.txt or check --data-dir option."

    assert args.out_path is not None, "--out-path must be specified."
    out_path = os.path.join(args.out_path, mol)

    if args.r_max is None:
        out_path = os.path.join(out_path, f"no_rmax")
        r_max = 6.0 ## default! but there is no edges
        r_edges = False
    else:
        out_path = os.path.join(out_path, f"rmax{args.r_max}")
        r_max = args.r_max
        r_edges = True
    os.makedirs(out_path, exist_ok=True)

    # Initialize feature extractor.
    a2g = AtomsToGraphs(
        max_neigh=100, ## rMD17 is molecule datasets, so there is no constraint on maximum number of neighbors.
        radius=r_max,
        r_energy=True, ## not args.test_data
        r_forces=True, ## not args.test_data
        r_fixed=False,
        r_distances=False,
        r_edges=r_edges, ## otf_graph can be False
    )

    print(f"Start preprocessing {mol}")

    ## convert npz to atoms (ase.atoms.Atoms)
    npz_data = np.load(npzfile)
    coords = npz_data["coords"]  # shape: (n_snapshot, n_atoms, 3)
    forces = npz_data["forces"]  # shape: (n_snapshot, n_atoms, 3)
    energies = npz_data["energies"]  # shape: (n_snapshot)
    forces = forces / kcalPerMol_FOR_1eV
    energies = energies / kcalPerMol_FOR_1eV

    nuclear_charges = npz_data["nuclear_charges"]  # shape: (n_atoms)
    #mol_symbols, mol_symbols_list = convert_symbols(nuclear_charges) # str

    charge_dict = {1:"H", 6:"C", 7:"N", 8:"O"}
    mol_symbols_list = [charge_dict[c] for c in nuclear_charges]

    n_snapshots = coords.shape[0]
    n_atoms = coords.shape[1]

    print(f"Number of atoms: {n_atoms}")
   
    ## split train/val/test
    np.random.seed(args.seed)
    index = np.random.permutation(np.arange(n_snapshots))
    train_index = index[:args.train_size]
    val_index = index[args.train_size:args.train_size+args.val_size]
    if args.test_size is None:
        test_index = index[args.train_size+args.val_size:]
    else:
        test_index = index[:-args.test_size]

    print(f"Number of snapshots: {len(index)}")
    print(f"-- Train: {len(train_index)}")
    print(f"-- Valid: {len(val_index)}")
    print(f"-- Test : {len(test_index)}")

    ## extract meta data which includes normalization statistics
    norm_stats = {
        "energy_mean": energies[train_index].mean(),
        "energy_std" : energies[train_index].std(),
        "force_mean" : 0.0, #forces[train_index].mean(), ## For atoms in a snapshot, the sum of forces should be zero.
        "force_std"  : forces[train_index].std(),
    }
    with open(os.path.join(out_path, "normalize_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=4)

    ## save train/val/test dataset
    for split_name, split_index in zip(["Trainset", "Validset", "Testset"], [train_index, val_index, test_index]):

        lmdbfile = os.path.join(out_path, split_name+".lmdb")
        db = lmdb.open(
            lmdbfile,
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
            lock=False,
        )

        for i, index in enumerate(tqdm(split_index)):
            ## 1) construct ase.Atoms 
            atoms = Atoms(symbols=mol_symbols_list,
                  positions=coords[index],
                  cell=250.0*np.identity(3), ## dummy cell (pbc = all False)
                  pbc=[False, False, False],
                  info={"energy": energies[index],
                        "free_energy": energies[index]},
                  #numbers=nuclear_charges,
                  )
            atoms.new_array("forces", forces[index])
            properties = {
                "energy": energies[index],
                "forces": forces[index],
            }
            calc = SinglePointCalculator(atoms, **properties)
            atoms.calc = calc
            
            ## 2) convert ase.Atoms into torch_geometric.Data
            data = a2g.convert(atoms)
            data.sid = 0
            data.fid = index
           
            ## 3) put torch_geometric.Data into LMDB
            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()

            # Save count of objects in lmdb.
            #txn = db.begin(write=True)
            #txn.put(f"length".encode("ascii"), pickle.dumps(len(traj_frames), protocol=-1))
            #txn.commit()
            ### --> It seems that length insertion is not needed..

        db.sync()
        db.close()

    print("Done")

    """
    with db.begin() as txn:
        for k, _ in txn.cursor():
            print(k)
    """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        help="Path of directory that includes *.npz of rMD17",
    )
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="If the cutoff radius is set, output LMDB files include edge index information, which means otf_graph in config files can be False.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=950,
        help="The number of train data (default: 950)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=50,
        help="The number of validation data (default: 50)",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="The number of test data (default: None, meaning that the remained samples will be used as test data.)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed for data shuffling",
    )
    args = parser.parse_args()

    for mol in molecules:
        write_npz_to_lmdb(args, mol)


if __name__ == "__main__":
    main()
