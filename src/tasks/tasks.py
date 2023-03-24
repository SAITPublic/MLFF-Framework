"""
This file will be deprecated
"""

import os

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask
from ocpmodels.common import distutils

from src.common.utils import bm_logging


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self):
        raise NotImplementedError("Predicion on test data with generating result files is not implemented")
        # assert (
        #     self.trainer.test_loader is not None
        # ), "Test dataset is required for making predictions"
        # assert self.config["checkpoint"]
        # results_file = "predictions"
        # self.trainer.predict(
        #     self.trainer.test_loader,
        #     results_file=results_file,
        # )