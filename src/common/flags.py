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

        # we modify argument options
        for action in self.parser._actions:
            if action.dest == "mode":
                action.choices = ["train", "fit-scale", "validate", "run-md", "evaluate"]
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
        # select molecule types in rMD17
        self.parser.add_argument(
            "--molecule", 
            default=None, 
            type=str, 
            help="For rMD17, molecule should be specified.",
        )
        # show progress bar (for evaluation)
        self.parser.add_argument(
            "--show-eval-progressbar", 
            default=False, 
            action="store_true", 
            help="Show a tqdm progressbar of calculating metrics (through validate())",
        )

    def _add_fit_scale_args(self):
        # some models need to generate scale files fitted to training data
        self.parser.add_argument(
            "--scale-path", 
            default="./",
            type=str, 
            help="Path where `model_scale.json` is saved. If None, the checkpoint including scaling factors will be generated."
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
        # checkpoint path which will be evaluated for metrics related to energy and force
        # -> this argument is already included (ocp/ocpmodels/common/flags.py)

        # dataset path
        self.parser.add_argument(
            "--validate-data",
            default=None,
            type=str,
            help="Data path to be evaluated in terms of energy and force metrics",
        )

        # batch size
        self.parser.add_argument(
            "--validate-batch-size",
            default=None,
            type=int,
            help="batch size (default : eval_batch_size specified in a config file)",
        )
        
    def _add_run_md_args(self):
        self.parser.add_argument(
            "--md-config-yml",
            default=None,
            type=str,
            help="Path to a config file listing MD simulation parameters",
        )
        
    def _add_evaluate_args(self):
        self.parser.add_argument(
            "--evaluation-metric",
            default=None,
            type=str,
            help="Evaluation metrics: energy_force (ef), distribution_functions (df), equation_of_state (eos), potential_energy_well (pe_well)",
        )
        self.parser.add_argument(
            "--evaluation-config-yml",
            default=None,
            type=str,
            help="Path to a config file listing evaluation configurations",
        )
        self.parser.add_argument(
            "--reference-trajectory",
            default=None,
            type=str,
            help="Path to a reference trajectory (.extxyz)",
        )
        

benchmark_flags = BenchmarkFlags()