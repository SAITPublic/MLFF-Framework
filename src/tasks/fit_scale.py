"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os
import math
import json
from itertools import islice

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel

from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask
from ocpmodels.modules.scaling import ScaleFactor

from src.common.utils import bm_logging


def _train_batch(trainer, batch):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=trainer.scaler is not None):
            trainer._forward(batch)


# reference : ocp/ocpmodels/modules/scaling/fit.py
@registry.register_task("fit-scale")
class FitScaleTask(BaseTask):
    def run(self):
        model = self.trainer.model
        while isinstance(model, (DistributedDataParallel, OCPDataParallel)):
            model = model.module
        model.eval()
        
        # set the data loader (train or valid)
        if self.config["data_type"] == "train":
            assert self.trainer.train_loader is not None
            data_loader = self.trainer.train_loader
        elif self.config["data_type"] == "valid":
            assert self.trainer.valid_loader is not None
            data_loader = self.trainer.valid_loader
        else:
            raise NotImplementedError(f"{self.config['data_type']} is not supported")
        bm_logging.info(f"Using {self.config['data_type']} data for fitting scaler modules")

        # initialize scale factors
        scale_factors = {}
        for name, module in model.named_modules():
            if isinstance(module, ScaleFactor):
                module.reset_()
                # GemNet-OC does not use module name
                # GemNet, PaiNN, GemNet-GP use module name
                module_name = name if module.name is None else module.name
                scale_factors[module_name] = module

        scale_factor_indices = {}
        idx = 0
        for name, module in scale_factors.items():
            def index_fn(name=name):
                nonlocal idx
                if name not in scale_factor_indices:
                    scale_factor_indices[name] = idx
                    idx += 1
            module.initialize_(index_fn = index_fn)
        
        # single pass through network (to find out the computation order of modules that include ScaleFactor)
        _train_batch(self.trainer, next(iter(data_loader)))

        # sort the scale factors by their computation order
        sorted_scale_factors = sorted(
            scale_factors.items(),
            key=lambda x: scale_factor_indices.get(x[0], math.inf),
        )
        bm_logging.info("Scale factors in order of their computation:")
        for name, _ in sorted_scale_factors:
            bm_logging.info(f"{name}: {scale_factor_indices[name]}")
        
        # loop over the scale factors in the computation order
        # and fit them one-by-one
        fit_results = {"comment": self.config["model"]["name"]}
        bm_logging.info(f"Start fitting (using {self.config['num_batches']} batches for each module)")
        for name, module in sorted_scale_factors:
            with module.fit_context_():
                for batch in islice(data_loader, self.config["num_batches"]):
                    _train_batch(self.trainer, batch)
                stats, ratio, value = module.fit_()
                fit_results[name] = value
                bm_logging.info(
                    f"[{name}] "
                    f"Var_in: {stats['variance_in']:.3f}, "
                    f"Var_out: {stats['variance_out']:.3f}, "
                    f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
                )
            assert module.fitted, f"{name} is not fitted"
        bm_logging.info("Done fitting")

        # save scaler files
        os.makedirs(self.config["scale_path"], exist_ok=True)
        if self.config["scale_file"] is None:
            scale_path = os.path.join(self.config["scale_path"], "model_scale.json")
        else:
            scale_path = os.path.join(self.config["scale_path"], self.config["scale_file"])
        with open(scale_path, "w", encoding="utf-8") as f:
            json.dump(fit_results, f, ensure_ascii=False, indent=4)
        bm_logging.info(f"The fitted scalers are saved at {scale_path}")

        