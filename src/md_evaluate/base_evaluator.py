"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os
import torch
import errno

from src.common.registry import md_evaluate_registry
from src.common.utils import bm_logging
from src.md_evaluate.calculator import BenchmarkCalculator


@md_evaluate_registry.register_md_evaluate("base_evaluator")
class BaseEvaluator:
    def __init__(self, config):
        if config.get("device"):
            self.device = torch.device(config["device"])
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        # load a config which was used to train the model
        self.config = config

        # load a checkpoint
        checkpoint_path = self.config.get("checkpoint", None)

        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(errno.ENOENT, "Checkpoint file not found", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu")

            # set a calculator using the loaded model
            self.calculator = BenchmarkCalculator(
                ckpt=ckpt,
                device=self.device,
            )
            
        self.logger = bm_logging

    def evaluate(self):
        """Derived evaluator classes should implement this function."""
        pass

    def simulate(self):
        """Simulator which is derived from BaseEvaluator should implement this function."""
        pass