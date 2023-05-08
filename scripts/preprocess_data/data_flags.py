"""
Written by byunggook.na (SAIT)

flags parsed by the preprocessing script
"""

import argparse


class DataFlags:
    def __init__(self):        
        self.parser = argparse.ArgumentParser(description="Data preprocessing")
        self.parser.add_argument(
            "--data",
            type=str,
            default=None,
            help="Path of a single data source file (.xyz or .extxyz)",
        )
        self.parser.add_argument(
            "--data-output-name",
            type=str,
            default=None,
            help="name of the single data output lmdb file (default policy is that its name is maintained)",
        )
        self.parser.add_argument(
            "--train-data",
            type=str,
            default=None,
            help="Path of a train data file (.xyz or .extxyz)",
        )
        self.parser.add_argument(
            "--train-data-output-name",
            type=str,
            default="train",
            help="name of the train data output lmdb file (default: train)",
        )
        self.parser.add_argument(
            "--valid-data",
            type=str,
            default=None,
            help="Path of a valid data file (.xyz or .extxyz)",
        )
        self.parser.add_argument(
            "--valid-data-output-name",
            type=str,
            default="valid",
            help="name of the valid data output lmdb file (default: valid)",
        )
        self.parser.add_argument(
            "--test-data",
            type=str,
            default=None,
            help="Path of a test data file (.xyz or .extxyz)",
        )
        self.parser.add_argument(
            "--test-data-output-name",
            type=str,
            default="test",
            help="name of the test data output lmdb file (default: test)",
        )
        self.parser.add_argument(
            "--out-path",
            type=str,
            default=None,
            help="Directory to save output data. If not given, the output data is saved at a parent directory where --data or --train-data exist",
        )
        self.parser.add_argument(
            "--r-max",
            type=float,
            default=None,
            help="If the cutoff radius is set, output LMDB files include edge index information, which means otf_graph in config files can be False.",
        )
        self.parser.add_argument(
            "--max-neighbors",
            type=int,
            default=None,
            help="The maximum number of neighbors",
        )
        self.parser.add_argument(
            "--save-normalization",
            type=bool,
            default=True,
            help="Save statistics obtained from train data for normalization"
        )
        self.parser.add_argument(
            "--energy-type",
            type=str,
            choices=["free_energy", "total_energy"],
            default="free_energy",
            help="Energy type used to calculate normalization information (default: free_energy)",
        )
        