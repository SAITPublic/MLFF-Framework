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
        if config.get("device"):
            self.device = torch.device(config["device"])
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        # load a config which was used to train the model
        self.config = config

        # load a checkpoint if given (derived evaluator "distribution_functions" does not require ckpt)
        checkpoint_path = self.config.get("checkpoint")
        self.calculator = None
        if checkpoint_path:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(errno.ENOENT, "Checkpoint file not found", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu")

            # set a calculator using the loaded model
            self.calculator = BenchmarkCalculator(
                ckpt=ckpt,    # note that scale_file path can be removed in the ckpt as it's not needed during inference
                device=self.device,
            )
            
        self.logger = bm_logging

    def evaluate(self):
        """Derived evaluator classes should implement this function."""
        pass

    def simulate(self):
        """Simulator which is derived from BaseEvaluator should implement this function."""
        pass