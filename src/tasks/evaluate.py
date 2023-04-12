"""
Written by byunggook.na
"""
import os

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask

from src.common.utils import bm_logging


@registry.register_task("evaluate")
class EvaluateTask(BaseTask):
    def setup(self, evaluator):
        self.evaluator = evaluator

    def run(self):
        self.evaluator.evaluate() 
        