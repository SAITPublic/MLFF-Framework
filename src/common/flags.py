import argparse
from ocpmodels.common.flags import Flags

class BenchmarkFlags(Flags):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Benchmark for machine learning force fields"
        )
        self.add_core_args()
        self._add_train_args()
        self._add_evaluate_args()

        for action in self.parser._actions:
            if action.dest == "mode":
                action.choices = ["train", "validate", "evaluate"]

    def _add_train_args(self):
        # to save checkpoints (only model state_dict) of every epoch
        self.parser.add_argument(
            "--save-ckpt-every-epoch", default=None, type=int, help="Save checkpoints at every given epoch",
        )
        # select molecule types in rMD17
        self.parser.add_argument(
            "--molecule", default=None, type=str, help="For rMD17, molecule should be specified.",
        )
        # show progress bar (for evaluation)
        self.parser.add_argument(
            "--show-eval-progressbar", default=False, action="store_true", help="Show a tqdm progressbar of evaluation",
        )


        # for prepare-checkpoint mode and fit.py
        self.parser.add_argument(
            "--prepare-ckpt-epoch", default=0, type=int, help="Training epoch for preparing a checkpoint which will be used to run fit.py to generate `scale_file.json`"
        )
        self.parser.add_argument(
            "--output-scale-file", default=None, type=str, help="Path to `scale_file.json` which will be used in training. If None, the checkpoint including scaling factors will be generated.")
        # print time and memory information during training
        self.parser.add_argument(
            "--view-time-mem-info", default=None, type=str, help="View information of time and memory during training. \nOptions: [change, all]. \n   change: view the information when memory size is changed.\n   all: view the information all the time"
        )
        
    def _add_evaluate_args(self):
        self.parser.add_argument(
            "--checkpoint-path", default=None, type=str, help="checkpoint path",
        )

benchmark_flags = BenchmarkFlags()