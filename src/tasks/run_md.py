"""
Written by byunggook.na
"""
import os

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask

from src.common.utils import bm_logging


@registry.register_task("run-md")
class RunMDTask(BaseTask):
    def setup(self, simulator):
        self.simulator = simulator

    def run(self):
        self.simulator.simulate()