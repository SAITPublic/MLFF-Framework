"""
Written by byunggook.na and heesun88.lee
"""
from src.common.registry import md_evaluate_registry
from src.md_evaluate.base_evaluator import BaseEvaluator


@md_evaluate_registry.register_md_evaluate("simulator")
class Simulator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        self.initial_structure = self.config["initial_structure"]
        self.output_trajectory = self.config["output_trajectory"]

    def simulate(self):
        """
        This body will be filled by heesun88.lee
        """
        print("run MD simulation")

        # To heesun88.lee
        # you can use self.calculator