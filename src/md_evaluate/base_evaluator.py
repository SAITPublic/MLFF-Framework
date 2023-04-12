"""
Written by byunggook.na
"""
import os

import torch

from src.common.registry import md_evaluate_registry
from src.common.utils import bm_logging # benchmark logging
from src.md_evaluate.calculator import BenchmarkCalculator


@md_evaluate_registry.register_md_evaluate("base_evaluator")
class BaseEvaluator:
    def __init__(self, config):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # load a config which was used to train the model
        self.config = config

        # load a checkpoint
        checkpoint_path = self.config["checkpoint"]
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(errno.ENOENT, "Checkpoint file not found", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # set a calculator using the loaded model
        self.calculator = BenchmarkCalculator(
            ckpt=ckpt,
            device=self.device,
        )

    def evaluate(self):
        """Derived evaluator classes should implement this function."""
        pass

    def simulate(self):
        """Simulator which is derived from BaseEvaluator should implement this function."""
        pass