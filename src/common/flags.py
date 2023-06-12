"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import argparse

from ocpmodels.common.flags import Flags


class BenchmarkFlags(Flags):
    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description="Benchmark for machine learning force fields")
        self.add_core_args() # OCP flags
        self._add_train_args()
        self._add_fit_scale_args()
        self._add_validate_args()
        self._add_run_md_args()
        self._add_evaluate_args()

        # we modify argument options defined in Flags of OCP
        for action in self.parser._actions:
            if action.dest == "mode":
                action.choices = ["fit-scale", "train", "validate", "run-md", "evaluate"]
            if action.dest == "config_yml":
                action.required = False

    def _add_train_args(self):
        # to save checkpoints (only model state_dict) of every epoch
        self.parser.add_argument(
            "--save-ckpt-every-epoch", 
            default=None, 
            type=int, 
            help="Save checkpoints at every given epoch",
        )
        # show progress bar (for evaluation)
        self.parser.add_argument(
            "--show-eval-progressbar", 
            default=False, 
            action="store_true", 
            help="Show a tqdm progressbar of calculating metrics (mainly used in the 'validate' mode)",
        )

    def _add_fit_scale_args(self):
        # some models need to generate scale files fitted to training data
        self.parser.add_argument(
            "--scale-path", 
            default="./",
            type=str, 
            help="Directory path where `model_scale.json` is saved. If None, the checkpoint including scaling factors will be generated."
        )
        # scale file name
        self.parser.add_argument(
            "--scale-file",
            default=None,
            type=str,
            help="Name of a scale file, which is .json",
        )
        # dataset to be used for fitting scalers
        self.parser.add_argument(
            "--data-type",
            default="train",
            type=str,
            help="Data type to be used for fitting scalers (train or valid)",
        )
        # num of batches to be used for fitting scalers
        self.parser.add_argument(
            "--num-batches",
            default=16,
            type=int,
            help="The number of batches to be used for fitting scalers",
        )

    def _add_validate_args(self):
        # dataset path
        self.parser.add_argument(
            "--validate-data",
            default=None,
            type=str,
            help="Data path to be evaluated in terms of energy and force metrics",
        )
        # batch size for validation
        self.parser.add_argument(
            "--validate-batch-size",
            default=None,
            type=int,
            help="batch size (default : eval_batch_size specified in a config file)",
        )
        
    def _add_run_md_args(self):
        # MD simulation configuration file
        self.parser.add_argument(
            "--md-config-yml",
            default=None,
            type=str,
            help="Path to a config file listing MD simulation parameters",
        )
        
    def _add_evaluate_args(self):
        # evaluation metrics including simulation indicators introduced in the paper
        self.parser.add_argument(
            "--evaluation-metric",
            default=None,
            type=str,
            help="Evaluation metrics: energy_force (ef), distribution_functions (df), equation_of_state (eos), potential_energy_curves (pe_curves)",
        )
        # evalution configuration file that includes structure information and simulation conditions
        self.parser.add_argument(
            "--evaluation-config-yml",
            default=None,
            type=str,
            help="Path to a config file listing evaluation configurations used for simulation indicators",
        )
        # reference trajectory to measure the model performance using energy and force errors
        self.parser.add_argument(
            "--reference-trajectory",
            default=None,
            type=str,
            help="Path to a reference trajectory (.extxyz) used in the 'energy_force' metric",
        )
        # save the predicted energy and forces
        self.parser.add_argument(
            "--save-ef",
            default=False,
            action="store_true", 
            help="Save a trajectory where energy and forces are predicted by a given MLFF model, given '--reference-trajectory'",
        )
        # print time profile
        self.parser.add_argument(
            "--measure-time",
            default=False,
            action="store_true", 
            help="Measure the average inference time per snapshot and per atom used in the 'energy_force' metric",
        )
        

benchmark_flags = BenchmarkFlags()