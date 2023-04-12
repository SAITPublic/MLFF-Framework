"""
Written by byunggook.na and heesun88.lee
"""
import ase
import ase.io

from src.common.registry import md_evaluate_registry
from src.md_evaluate.base_evaluator import BaseEvaluator


@md_evaluate_registry.register_md_evaluate("pew")
class PEWEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        self.ref_traj_path = self.config["reference_trajectory"]
        self.pred_traj_path = self.config["generated_trajectory"]

    def evaluate(self):
        """
        This body will be filled by heesun88.lee
        """
        print("Run PEW evaluation")