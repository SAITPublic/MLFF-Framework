"""
Written by byunggook.na and heesun88.lee
"""

import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch

from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import load_state_dict
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer

from src.common.registry import evaluator_registry
from src.modules.calculator import BenchmarkCalculator


@evaluator_registry.register_evaluator("base_evaluator")
class BaseEvaluator(ABC):
    def __init__(self, config):
        assert config is not None

        self.config = self._parse_config(config)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self._inititiate()

        # TODO (byunggook): refactoring.. because there exists trainser.config in checkpoint

    def _parse_config(self, config):
        # TODO (heesun)
        # Parse the config and extract arguments required to evalute MLFF models with MD simulations.
        # Command arguments (which are not listed in a configuration yaml file) can be handled.
        # reference: _parse_config() in src/trainers/base_trainer.py

        # A model_path is given through an argument of main.py 
        #   (i.e., python main.py --model-path MY_MODEL)
        # I think it is better to handle various models with the same evaluation configuration file
        evaluator_config = {
            "task": config["task"],
            "model_name": config["model"].pop("name"),
            "model_attributes": config["model"],
            "cmd": {
                "checkpoint_path": config["checkpoint_path"],
            },
        }
        # specify dataset path
        dataset = config["dataset"]
        if isinstance(dataset, list):
            if len(dataset) > 0:
                evaluator_config["dataset"] = dataset[0]
        elif isinstance(dataset, dict):
            evaluator_config["dataset"] = dataset.get("train", None)
        else:
            evaluator_config["dataset"] = dataset

        return evaluator_config

    def _inititiate(self):
        # TODO (heesun)
        # initiate components
        # for examples, calculator, ase simulation instance, MDlogger, ...
        # reference: _initiate() in src/trainers/base_trainer.py
        self._set_model()
        self._set_calculator()

    def _set_model(self):
        checkpoint_path = self.config["cmd"]["checkpoint_path"]
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(errno.ENOENT, "Checkpoint file not found", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, BaseModel):
            # Case 1. Load a checkpoint as a model class
            # not complete!
            raise NotImplementedError("Not implemented ... because of normalizers maintained in a trainer class")
            # self.model = ckpt.to(self.device)
        elif isinstance(ckpt, dict):
            # Case 2. Load a checkpoint as a dictionary that includes state_dict() and other auxiliary information
            # It requires the corresponding configuration file of the model
            model_class = registry.get_model_class(self.config["model_name"])
            self.model = model_class(
                num_atoms = None, # not used
                bond_feat_dim = None, # not used
                num_targets = 1, # always 1
                **self.config["model_attributes"],
            )
            model_state_dict = OrderedDict()
            for key, val in ckpt["state_dict"].items():
                if key.startswith("module."):
                    k = key[7:]
                else:
                    k = key
                model_state_dict[k] = val
            
            load_state_dict(module=self.model, state_dict=model_state_dict, strict=True)
            self.model = self.model.to(self.device)
            # incompat_keys = self.model.load_state_dict(ckpt["state_dict"])
            # _report_incompat_keys(self.model, incompat_keys, strict=True)

            # normalizers
            self.normalizers = {}
            if self.config["dataset"].get("normalize_labels", False):
                for key in ["target", "grad_target"]:
                    self.normalizers[key] = Normalizer()
                    self.normalizers[key].load_state_dict(ckpt["normalizers"][key])

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

    def _set_calculator(self):
        self.calculator = BenchmarkCalculator(
            config=self.config,
            model=self.model,
            normalizers=self.normalizers,
        )

    @abstractmethod
    def evaluate(self):
        """Derivced evaluator classes should implement this function."""
        # this function is main body of our evaluation based on simulation metrics