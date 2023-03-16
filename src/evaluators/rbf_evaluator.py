"""
Written by byunggook.na and heesun88.lee
"""

from src.common.registry import evaluator_registry
from src.evaluators.base_evaluator import BaseEvaluator


@evaluator_registry.register_evaluator("rbf")
class RBFEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config=config)

    def evaluate(self):
        print("Run RBF evaluation")