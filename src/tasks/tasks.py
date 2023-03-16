"""
Copied from ocp.ocpmodels.trainers.forces_trainer.py
Modifications:
1) modify argument related to showing progress bars
"""

import os

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask
from ocpmodels.common import distutils

from src.common.utils import bm_logging


@registry.register_task("train")
class TrainTask(BaseTask):
    def run(self):
        try:
            self.trainer.train()

            # save the model as a class with trained parameters
            self.trainer.save_model_as_class()

        except RuntimeError as e:
            self._process_error(e)
            e_str = str(e)
            if (
                "find_unused_parameters" in e_str
                and "torch.nn.parallel.DistributedDataParallel" in e_str
            ):
                for name, parameter in self.trainer.model.named_parameters():
                    if parameter.requires_grad and parameter.grad is None:
                        bm_logging.warning(
                            f"Parameter {name} has no gradient. Consider removing it from the model."
                        )
            raise e


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self):
        # Note that the results won't be precise on multi GPUs due to padding of extra images (although the difference should be minor)
        assert (
            self.trainer.val_loader is not None
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        bm_logging.info("Validate energy and force for validation data ... ")
        self.trainer.validate(split="val")
        if self.trainer.test_loader is not None:
            bm_logging.info("Validate energy and force for test data ... ")
            self.trainer.validate(split="test")


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


@registry.register_task("evaluate")
class EvaluateTask(BaseTask):
    def setup(self, evaluator):
        self.evaluator = evaluator

    def run(self):
        ## TODO: for each metric script, script body is located in run() 
        self.evaluator.evaluate() 
        