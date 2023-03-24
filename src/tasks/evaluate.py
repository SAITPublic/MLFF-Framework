import os

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask

from src.common.utils import bm_logging


@registry.register_task("evaluate")
class EvaluateTask(BaseTask):
    def setup(self, evaluator):
        self.evaluator = evaluator

    def run(self):
        ## TODO: for each metric script, script body is located in run() 
        self.evaluator.evaluate() 
        