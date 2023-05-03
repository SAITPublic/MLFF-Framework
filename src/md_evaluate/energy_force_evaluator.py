"""
Written by byunggook.na
"""
import torch
import numpy as np
import math

from prettytable import PrettyTable
from pathlib import Path

import ase
import ase.io

from src.common.utils import bm_logging # benchmark logging
from src.common.registry import md_evaluate_registry
from src.modules.metric_evaluator import MetricEvaluator
from src.md_evaluate.base_evaluator import BaseEvaluator


@md_evaluate_registry.register_md_evaluate("ef")
@md_evaluate_registry.register_md_evaluate("energy_force")
class EnergyForceEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        self.ref_traj_path = self.config["reference_trajectory"]
        assert self.ref_traj_path is not None, "--reference-trajectory should be given."
        self.display_meV = True

        self.metric_evaluator = MetricEvaluator(
            task="s2ef", 
            task_metrics=["energy_per_atom_mae", "energy_per_atom_mse", "forces_mae", "forces_mse"],
            device=self.device,
        )

    def evaluate(self):
        ref_traj = ase.io.read(self.ref_traj_path, index=":", format="extxyz")

        bm_logging.info("Start evaluation to predict energy and forces")
        metrics = {}
        for atoms in ref_traj:
            target = {
                "energy": torch.tensor(atoms.info["energy"], dtype=torch.float32),
                "forces": torch.tensor(atoms.arrays["forces"]),
                "natoms": torch.tensor(atoms.arrays["forces"].shape[0], dtype=torch.long),
            }
            self.calculator.calculate(atoms=atoms)
            pred = {
                "energy": torch.tensor(self.calculator.results["energy"], dtype=torch.float32),
                "forces": torch.tensor(self.calculator.results["forces"]),
                "natoms": target["natoms"],
            }
            metrics = self.metric_evaluator.eval(pred, target, prev_metrics=metrics)

        table = PrettyTable()
        table.field_names = ["dataset"] + [metric_name for metric_name in self.metric_evaluator.metric_fn]
        table_row_metrics = [Path(self.ref_traj_path).name]
        for metric_name in self.metric_evaluator.metric_fn:
            if self.display_meV and "mae" in metric_name:
                table_row_metrics.append(f"{metrics[metric_name]['metric'] * 1000:.1f}")
            elif self.display_meV and "mse" in metric_name:
                # mse
                # table_row_metrics.append(f"{metrics[metric_name]['metric'] * 1000000:.1f}") 
                # rmse
                table_row_metrics.append(f"{math.sqrt(metrics[metric_name]['metric']) * 1000:.1f}")
            else:
                table_row_metrics.append(f"{metrics[metric_name]['metric']:.1f}")
        table.add_row(table_row_metrics)
        bm_logging.info(f"\n{table}")
        

        


