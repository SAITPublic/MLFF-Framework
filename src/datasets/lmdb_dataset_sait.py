"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import bisect
import logging
import math
import pickle
import random
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import pyg2_data_transform

from src.common.utils import bm_logging


def random_sample(lst, num_samples=5):
    indices = np.random.choice(len(lst), num_samples, replace=False)
    sampled_elements = [lst[i] for i in indices]
    return sampled_elements


@registry.register_dataset("lmdb_sait")
class LmdbDatasetSAIT(Dataset):
    def __init__(self, config, transform=None):
        super(LmdbDatasetSAIT, self).__init__()
        self.config = config

        # listup lmdb files
        self.db_paths = []
        for src_info in self.config:
            src_path = Path(src_info["src"])
            if src_path.is_dir():
                # directory including lmdb file(s)
                file_paths = sorted(src_path.glob("*.lmdb"))
                assert len(file_paths) > 0, f"No LMDBs found in '{src_path}'"
                if "sampled_ratio" in src_info.keys():
                    file_paths = [[f, src_info["sampled_ratio"]] for f in file_paths]
                self.db_paths += file_paths
            else:
                # a single lmdb file
                assert src_path.exists(), f"No LMDB found at '{src_path}'"
                if "sampled_ratio" in src_info.keys():
                    src_path = [src_path, src_info["sampled_ratio"]]
                self.db_paths.append(src_path)

        # setup databases
        self._keys, self.envs = [], []
        for db_path in self.db_paths:
            sampled_ratio = None
            if isinstance(db_path, list):
                sampled_ratio = db_path[1]
                db_path = db_path[0]
            env = self.connect_db(db_path)
            self.envs.append(env)
            keys = [f"{j}".encode("ascii") for j in range(env.stat()["entries"])]
            if sampled_ratio is not None:
                keys = random_sample(keys, num_samples=int(len(keys)*sampled_ratio))
            self._keys.append(keys)

            bm_logging.info(f" - {db_path.resolve()}")

        # append all the lmdb files by managing keys
        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Figure out which db this should be indexed from
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Retrieve data sample
        datapoint_pickled = self.envs[db_idx].begin().get(self._keys[db_idx][el_idx])
        data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
        
        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        for env in self.envs:
            env.close()