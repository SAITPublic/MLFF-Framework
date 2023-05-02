"""
Written by byunggook.na and heesun88.lee
"""
import numpy as np
import pandas as pd
import copy
import json
from collections import defaultdict
import os
from math import log10, floor, isnan
from pathlib import Path
import matplotlib.pyplot as plt
from ase import io
from ase.eos import EquationOfState

from src.common.registry import md_evaluate_registry
from src.md_evaluate.base_evaluator import BaseEvaluator


@md_evaluate_registry.register_md_evaluate("eos")
class EoSEvaluator(BaseEvaluator):

    def calculate_eos_error(self, eos_fit_ref, eos_fit_mlff, save_res=True):
        eos_fit_ref["B_error_percentage"] = np.nan
        eos_fit_ref["v0_error_percentage"] = np.nan
        eos_fit_ref["e0_error_percentage"] = np.nan

        eos_fit_mlff["B_error_percentage"] = \
            (eos_fit_mlff["B (GPa)"] - eos_fit_ref["B (GPa)"]) / \
            eos_fit_ref["B (GPa)"]*100.0
        eos_fit_mlff["v0_error_percentage"] = \
            (eos_fit_mlff["v0"] - eos_fit_ref["v0"])/eos_fit_ref["v0"]*100.0
        eos_fit_mlff["e0_error_percentage"] = \
            (eos_fit_mlff["e0"] - eos_fit_ref["e0"])/eos_fit_ref["e0"]*100.0

        fit_res_dict_all = {"reference": eos_fit_ref}
        fit_res_dict_all.update({"model": eos_fit_mlff})

        res_df = pd.DataFrame.from_dict(fit_res_dict_all)
        if save_res:
            res_df.to_csv(
                Path(self.config["res_out_dir"]) / "eos_error_metrics.csv")

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            self.logger.debug(res_df.applymap(lambda x: round(x, 3 - int(floor(log10(abs(x))))) if not isnan(x) else x))

    def plot_eos(self, df_vasp, df_mlff, fig_out_path, save_res):
        plt.figure()
        ax = df_vasp.plot(x="volume", y="e-e0", marker=".")
        df_mlff.plot(x="volume", y="e-e0", marker=".", ax=ax)
        label_list = ["dft", "model"]

        ax.set_xlabel('volume (ang.^3)')
        ax.set_ylabel('E-E0 (eV)')
        ax.legend(label_list)
        # plt.show()
        if save_res:
            plt.savefig(fig_out_path)
            self.logger.debug("Figure to compare EoS was saved in {}".format(fig_out_path))
        # plt.close('all')

    @staticmethod
    def save_eos_fit_res(out_path, eos_fit_dict):
        with open(out_path, "w") as f:
            json.dump(eos_fit_dict, f, indent=4)

    def load_reference_results(self, file_path, scale_factors, eos_type):
        data = defaultdict(list)
        with open(file_path, "r") as f:
            content = f.read().splitlines()
            for i in range(len(content)):
                data["scale_factor"].append(float(content[i].split()[0]))
                data["volume"].append(float(content[i].split()[1]))
                data["PE"].append(float(content[i].split()[2]))

        df = pd.DataFrame(data)
        range_mask = (df["scale_factor"] >= scale_factors[0]) & \
            (df["scale_factor"] <= scale_factors[-1])
        df = df[range_mask]
        self.logger.debug("Number of data points to consider for eos (reference): {}".format(
            df.shape[0]))

        eos_fit_dict = self.calculate_eos_fit(df, eos_type)
        df["e-e0"] = df["PE"] - eos_fit_dict["e0"]
        return df, eos_fit_dict

    @staticmethod
    def calculate_eos_fit(df, eos_type):
        eos = EquationOfState(df["volume"], df["PE"], eos=eos_type)
        v0, e0, Bm = eos.fit()
        eos_fit_dict = {"v0": v0, "e0": e0, "B (GPa)": Bm*160.2176621}
        return eos_fit_dict

    def evaluate(self):
        atoms_no_scale = io.read(self.config["ref_structure"]["path"],
                                 format=self.config["ref_structure"].get("format"))
        original_cell = atoms_no_scale.get_cell().__array__()
        atoms_no_scale.calc = self.calculator

        data = defaultdict(list)
        scale_factors = np.arange(self.config["scale_factors"]["start"],
                                  self.config["scale_factors"]["end"],
                                  self.config["scale_factors"]["interval"])

        for scale_factor in scale_factors:
            scaled_cell = original_cell * scale_factor
            # check if the calculator of atoms_no_scale is also copied
            atoms = copy.deepcopy(atoms_no_scale)
            atoms.set_cell(scaled_cell, scale_atoms=True)

            pe = atoms.get_potential_energy()
            volume = atoms.get_volume()

            data["scale_factor"].append(scale_factor)
            data["volume"].append(volume)
            data["PE"].append(pe)

        df_mlff = pd.DataFrame(data, columns=["scale_factor", 'volume', "PE"])
        df_mlff.set_index("scale_factor", inplace=True)

        eos_fit_mlff = EoSEvaluator.calculate_eos_fit(
            df_mlff, self.config["eos_type"])

        os.makedirs(Path(self.config["res_out_dir"]), exist_ok=True)
        df_mlff.to_csv(
            Path(self.config["res_out_dir"]) / "volume_energy_relation.csv")

        eos_res_name = "eos_res_{}.txt".format(self.config["eos_type"])
        save_path_eos = Path(self.config["res_out_dir"]) / eos_res_name
        EoSEvaluator.save_eos_fit_res(save_path_eos, eos_fit_mlff)
        df_mlff["e-e0"] = df_mlff["PE"] - eos_fit_mlff["e0"]

        ref_v_e_path = Path(self.config["reference_result"]["dir"]) / \
            self.config["reference_result"]["volume_energy_fname"]
        df_ref, eos_fit_ref = self.load_reference_results(
            ref_v_e_path,
            scale_factors,
            self.config["eos_type"]
        )

        if self.config["reference_result"]["save_eos_fit"]:
            save_path_eos_ref = save_path_eos = Path(
                self.config["reference_result"]["dir"]) / eos_res_name
            EoSEvaluator.save_eos_fit_res(save_path_eos_ref, eos_fit_ref)

        self.calculate_eos_error(eos_fit_ref, eos_fit_mlff, save_res=True)

        fig_out_dir = Path(
            self.config["res_out_dir"]) / self.config["res_fig_name"]
        self.plot_eos(df_ref,
                      df_mlff,
                      fig_out_dir,
                      save_res=True)
