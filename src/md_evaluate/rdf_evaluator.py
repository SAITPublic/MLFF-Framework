"""
Written by byunggook.na and heesun88.lee
"""
import torch
import numpy as np

import ase
import ase.io

from src.common.registry import md_evaluate_registry
from src.md_evaluate.base_evaluator import BaseEvaluator


@md_evaluate_registry.register_md_evaluate("rdf")
class RDFEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        
        self.ref_traj_path = self.config["reference_trajectory"]
        self.pred_traj_path = self.config["generated_trajectory"]

    def evaluate(self):
        """
        This body will be filled by heesun88.lee
        """
        print("Run RDF evaluation")

        ## ADF is included this evaluator ???

        # test
        ref_traj = ase.io.read(self.ref_traj_path, index=":", format="extxyz")
        
        gt_energy = []
        pred_energy = []
        gt_forces = []
        pred_forces = []
        num_atoms = []
        
        energy_mae = []

        print("start ... ")
        for atoms in ref_traj:
            # atoms.set_calculator(self.calculator)

            gt_energy.append(atoms.info["energy"])
            gt_forces.append(torch.tensor(atoms.arrays["forces"]))
            num_atoms.append(atoms.arrays["forces"].shape[0])

            self.calculator.calculate(atoms=atoms)

            # print("energy", atoms.info["energy"])
            # print("forces", atoms.arrays["forces"])
            # atoms.get_calculator().calculate()
            # print("pred energy", atoms.calculator.results["energy"])
            # print("pred forces", atoms.calculator.results["forces"])

            pred_energy.append(self.calculator.results["energy"])
            pred_forces.append(torch.tensor(self.calculator.results["forces"]))

            # energy_mae.append(torch.abs(gt_energy[-1] - pred_energy[-1])/gt_forces[-1].shape[0])
            energy_mae.append(np.abs(gt_energy[-1] - pred_energy[-1])/gt_forces[-1].shape[0])

        print("energy per atom mae")
        print(torch.tensor(energy_mae).mean().item())

        print("force per dim mae")
        print(torch.abs(torch.stack(gt_forces) - torch.stack(pred_forces)).mean().item())
        
        

        


