"""
Written by byunggook.na 
(reference: ocp/scripts/preprocess_ef.py)

Creates LMDB files with extracted graph features from provided *.npz files of rMD17
for the S2EF task.

rMD17 data details:
'nuclear_charges' : The nuclear charges for the molecule
'coords' : The coordinates for each conformation (in units of ångstrom)
'energies' : The total energy of each conformation (in units of kcal/mol)
'forces' : The cartesian forces of each conformation (in units of kcal/mol/ångstrom)
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
    assert os.path.isfile(npzfile), f"{npzfile} does not exist."

    out_path = os.path.join(args.out_path, mol)

    if args.r_max is None:
        out_path = os.path.join(out_path, f"atom_cloud")
        r_edges = False
    else:
        out_path = os.path.join(out_path, f"atom_graph_rmax{args.r_max}")
        r_edges = True

    if args.sampling_step is None:
        out_path = out_path + "_uniform"
    else:
        out_path = out_path + f"_sampling_step_{args.sampling_step}"
    os.makedirs(out_path, exist_ok=True)

    # Initialize feature extractor.
    # rMD17 consists of small molecules which have under 100 atoms, 
    # so actually there is no constraint on maximum number of neighbors.
    a2g = AtomsToGraphs(
        max_neigh=100, 
        radius=args.r_max,
        r_energy=True, 
        r_forces=True, 
        r_fixed=False,
        r_distances=False,
        r_edges=r_edges, 
    )

    print(f"Start preprocessing {mol}")

    # convert npz to atoms (ase.atoms.Atoms)
    npz_data = np.load(npzfile)
    coords = npz_data["coords"]  # shape: (n_snapshot, n_atoms, 3)
    forces = npz_data["forces"]  # shape: (n_snapshot, n_atoms, 3)
    energies = npz_data["energies"]  # shape: (n_snapshot)
    forces = forces / kcalPerMol_FOR_1eV
    energies = energies / kcalPerMol_FOR_1eV

    nuclear_charges = npz_data["nuclear_charges"]  # shape: (n_atoms)
    charge_dict = {1:"H", 6:"C", 7:"N", 8:"O"}
    mol_symbols_list = [charge_dict[c] for c in nuclear_charges]

    n_snapshots = coords.shape[0]
    n_atoms = coords.shape[1]

    print(f"Number of atoms: {n_atoms}")
   
    # split train/val/test
    if args.sampling_step is None:
        # uniform sampling (random sampling)
        np.random.seed(args.seed)
        index = np.random.permutation(np.arange(n_snapshots))
        train_index = index[:args.train_size]
        val_index = index[args.train_size:args.train_size+args.val_size]
        if args.test_size is None:
            test_index = index[args.train_size+args.val_size:]
        else:
            test_index = index[:-args.test_size]
    else:
        # sampling with the constant step
        start_index = np.random.randint(n_snapshots - (args.train_size + args.val_size - 1) * args.sampling_step)
        sampled_index = np.arange(start_index, n_snapshots, args.sampling_step)
        if sampled_index[-1] >= n_snapshots:
            raise ValueError("Indices of data which will be sampled are larger than the number of the total data")
        sampled_index = np.random.permutation(sampled_index)
        train_index = sampled_index[:args.train_size]
        val_index = sampled_index[args.train_size:]
        test_index = np.array([i for i in range(n_snapshots) if (i-start_index)%args.sampling_step != 0])
        if args.test_size is not None:
            test_index = np.random.choice(test_index, args.test_size)

    print(f"Number of snapshots: {n_snapshots}")
    print(f"-- Train: {len(train_index)}")
    print(f"-- Valid: {len(val_index)}")
    print(f"-- Test : {len(test_index)}")

    indices_dict = {'train': sorted(train_index), 'valid': sorted(val_index), 'test': sorted(test_index)}

    # extract meta data which includes normalization statistics
    norm_stats = {
        "energy_mean": energies[train_index].mean(),
        "energy_std" : energies[train_index].std(),
        "force_mean" : 0.0, # The sum of forces in each snapshots should be zero, so the mean should be also zero.
        "force_std"  : forces[train_index].std(),
    }
    with open(os.path.join(out_path, "normalize_stats.json"), "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=4)

    # save train/val/test dataset
    for split_name, split_index in indices_dict.items():
        lmdbfile = os.path.join(out_path, f"{split_name}.lmdb")
        db = lmdb.open(
            lmdbfile,
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
            lock=False,
        )

        for i, index in enumerate(tqdm(split_index)):
            # 1) construct ase.Atoms 
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
            
            # 2) convert ase.Atoms into torch_geometric.Data
            data = a2g.convert(atoms)
            data.sid = 0
            data.fid = index
           
            # 3) put torch_geometric.Data into LMDB
            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()

        db.sync()
        db.close()

        # save the index list 
        indexfile = os.path.join(out_path, f"{split_name}_index.csv")
        with open(indexfile, 'w', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            for i, idx in enumerate(split_index):
                wr.writerow([idx])

    print("Done")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        help="Path of directory that includes *.npz of rMD17",
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
    parser.add_argument(
        "--sampling-step",
        type=int,
        default=None,
        help="Sampling step, at every which data is sampled as the train and validation data (default: None, meaning random sampling)"
    )
    args = parser.parse_args()

    for mol in molecules:
        write_npz_to_lmdb(args, mol)


if __name__ == "__main__":
    main()
