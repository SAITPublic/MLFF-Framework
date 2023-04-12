import os

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask

from src.common.utils import bm_logging


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self):
        pass