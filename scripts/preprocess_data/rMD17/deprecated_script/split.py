
import os
import sys
import numpy as np
import ase
import ase.io

import argparse

molecule_type = ["aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde", "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"]


parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type=int, default=950)
parser.add_argument('--val_size', type=int, default=50)
parser.add_argument('--data_dir', type=str, default="/mnt/nas/DB/rmd17/xyz_data/")
parser.add_argument('--split_index_dir', type=str, default="./")
args = parser.parse_args()

def main():
    for mol in molecule_type:
        data = ase.io.read(os.path.join(args.data_dir, f"rmd17_{mol}.xyz"), format="extxyz", index=":")
        
        num_snapshots = len(data)
        index = np.arange(num_snapshots)
        index = np.random.permutation(index)
        
        f = open(os.path.join(args.split_index_dir, f"rmd17_{mol}_train.txt"), 'w')
        for i in index[:args.train_size]:
            f.write(f"{i}\n")
        f.close()

        f = open(os.path.join(args.split_index_dir, f"rmd17_{mol}_val.txt"), 'w')
        for i in index[args.train_size:args.train_size+args.val_size]:
            f.write(f"{i}\n")
        f.close()

        f = open(os.path.join(args.split_index_dir, f"rmd17_{mol}_test.txt"), 'w')
        for i in index[args.train_size+args.val_size:]:
            f.write(f"{i}\n")
        f.close()

if __name__ == '__main__':
    main()
