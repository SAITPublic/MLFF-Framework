"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

from pathlib import Path
from scipy import interpolate, optimize
from collections import defaultdict

from ase import io

from src.common.utils import calc_error_metric
from src.common.registry import md_evaluate_registry
from src.md_evaluate.base_evaluator import BaseEvaluator


@md_evaluate_registry.register_md_evaluate("pe_well")
@md_evaluate_registry.register_md_evaluate("potential_energy_well")
class PEWEvaluator(BaseEvaluator):

    def calculate_pe_well_error_metric(self, df_ref_in, df_mlff_in, res_out_path):
        df_ref = df_ref_in.rename(columns={"PE": "PE_ref"})
        df_mlff = df_mlff_in.rename(columns={"PE": "PE_mlff"})

        df_combined = pd.concat([df_ref, df_mlff], axis=1, join="inner")
        dist_array = df_combined.index.values

        f_ref = interpolate.interp1d(dist_array, df_combined["PE_ref"].values,
                                     bounds_error=False)
        min_out = optimize.fmin(f_ref, 2.)
        r0_ref = min_out[0]
        E0_ref = f_ref(r0_ref)
        mask = (df_combined["PE_ref"] <= 0)

        f_mlff = interpolate.interp1d(dist_array, df_combined["PE_mlff"].values,
                                      bounds_error=False)
        min_out_1 = optimize.fmin(f_mlff, dist_array[0])
        min_out_2 = optimize.fmin(f_mlff, df_combined[mask].index.values[0])
        min_out_3 = optimize.fmin(f_mlff, r0_ref)
        min_out_4 = optimize.fmin(f_mlff, dist_array[-1])

        min_out_list = [min_out_1, min_out_2, min_out_3, min_out_4]
        min_dist_list = list(set([round(min_out[0], 4) for min_out in min_out_list]))
        min_PE_list = [float(f_mlff(dist)) for dist in min_dist_list]

        r0_mlff = min(min_dist_list, key=lambda x: abs(x-r0_ref))
        E0_mlff = float(f_mlff(r0_mlff))

        self.logger.info("[mlff model] r0: {}, E0: {}".format(r0_mlff, E0_mlff))
        self.logger.info("[mlff model] min_dist_list: {}, min_PE_list: {}".format(
            min_dist_list, min_PE_list))

        pe_mae = calc_error_metric(df_combined["PE_mlff"].values,
                                   df_combined["PE_ref"].values,
                                   "mae")
        pe_rmse = calc_error_metric(df_combined["PE_mlff"].values,
                                    df_combined["PE_ref"].values,
                                    "rmse")

        dist_array_trimmed = df_combined[mask].index.values
        pe_mae_trimmed = calc_error_metric(df_combined[mask]["PE_mlff"].values,
                                           df_combined[mask]["PE_ref"].values,
                                           "mae")
        pe_rmse_trimmed = calc_error_metric(df_combined[mask]["PE_mlff"].values,
                                            df_combined[mask]["PE_ref"].values,
                                            "rmse")

        pe_deriv_ref = np.gradient(df_combined["PE_ref"].values, dist_array)
        pe_deriv_mlff = np.gradient(df_combined["PE_mlff"].values, dist_array)

        pe_deriv_mae = calc_error_metric(pe_deriv_mlff, pe_deriv_ref, "mae")
        pe_deriv_rmse = calc_error_metric(pe_deriv_mlff, pe_deriv_ref, "rmse")
        pe_deriv_mae_trimmed = calc_error_metric(pe_deriv_mlff[mask], pe_deriv_ref[mask], "mae")
        pe_deriv_rmse_trimmed = calc_error_metric(pe_deriv_mlff[mask], pe_deriv_ref[mask], "rmse")

        with open(res_out_path, 'w') as f:
            f.write(f"Interatomic distances considered for interpolation: {dist_array.tolist()} \n")
            f.write(f"Interatomic distances considered for PE and PE' error calc. trimmed: {dist_array_trimmed.tolist()} \n")
            f.write(f"1) interatomic distances that are local minima: {min_dist_list}, \n \
                    r0 (local minimum closest to ref r0): {r0_mlff}, \n \
                    error in r0: {r0_mlff-r0_ref} \n"
            )
            f.write(f"2) PE at local minima: {min_PE_list}, \n \
                    E0 (PE at local minimum that is closet to ref r0): {E0_mlff}, \n \
                    error in E0: {E0_mlff - E0_ref} \n"
            )
            f.write(f"3) PE error - MAE: {pe_mae}, RMSE: {pe_rmse} \n")
            f.write(f"3') PE error trimmed - MAE: {pe_mae_trimmed}, RMSE: {pe_rmse_trimmed} \n")
            f.write(f"4) PE derivative (F) error - MAE: {pe_deriv_mae}, RMSE: {pe_deriv_rmse}\n")
            f.write(f"4') PE derivative (F) error trimmed - MAE: {pe_deriv_mae_trimmed}, RMSE: {pe_deriv_rmse_trimmed}\n")

    def generate_comparison_plot(self, df_ref, df_mlff, fig_out_path, save_res):
        plt.figure()
        ax = df_ref.plot(y="PE", use_index=True)
        df_mlff.plot(y="PE", use_index=True, ax=ax)
        label_list = ["reference", "mlff"]

        ax.set_xlabel('distance (ang.)')
        ax.set_ylabel('PE (eV)')
        ax.legend(label_list)
        plt.axhline(y=0, color='k', linestyle='-')
        if save_res:
            plt.savefig(fig_out_path)
            self.logger.info(f"Figure to compare PE Well was saved in {fig_out_path}")

    def load_reference_results(self, file_path):
        data_ref = defaultdict(list)
        with open(file_path, "r") as f:
            content = f.read().splitlines()
            for i in range(len(content)):
                data_ref["dist"].append(float(content[i].split()[0]))
                data_ref["PE"].append(float(content[i].split()[1]))

        df_ref = pd.DataFrame(data_ref)
        df_ref.set_index("dist", inplace=True)
        self.logger.info(f"Number of data points for pe_well (reference): {df_ref.shape[0]}")
        return df_ref

    @staticmethod
    def construct_range_array(range_dict):
        n_valid_decimal = 5
        return np.arange(range_dict["start"],
                         range_dict["end"],
                         range_dict["interval"]).round(n_valid_decimal)  # round to truncate spurious tailing numbers

    def evaluate(self):
        calc_failure_msg_template = "PE calculation for unit structure '{}', \
            distance (or angle) {} was failed. Run will continue."

        for unit_structure_dict in self.config["unit_structures"]:
            structure_name = unit_structure_dict["name"]

            if unit_structure_dict.get("interatomic_distances"):
                atoms = io.read(unit_structure_dict["path"],
                                format=unit_structure_dict["format"])
                assert len(atoms) == 2, "For an enetry of unit_structures, \
                if 'interatomic_distances' field is provided, structure file should have only 2 atoms!"
                atoms.set_pbc(False)
                atoms.calc = self.calculator
                data = defaultdict(list)
                for dist in PEWEvaluator.construct_range_array(unit_structure_dict["interatomic_distances"]):
                    try:
                        atoms.set_positions([(0, 0, 0), (dist, 0, 0)])
                        pe = atoms.get_potential_energy()
                    except:
                        print(calc_failure_msg_template.format(
                            structure_name, dist))
                        print(traceback.format_exc())
                        pe = np.nan

                    print("dist: {}, PE: {}".format(dist, pe))
                    data["dist"].append(dist)
                    data["PE"].append(pe)

            elif unit_structure_dict.get("filename_scale_suffixes"):
                data = defaultdict(list)
                for scale_suffix in PEWEvaluator.construct_range_array(unit_structure_dict["filename_scale_suffixes"]):
                    structure_path = "{}_{}".format(unit_structure_dict["path"],
                                                    str(scale_suffix))
                    atoms = io.read(structure_path,
                                    format=unit_structure_dict["format"])
                    atoms.set_pbc(False)
                    atoms.calc = self.calculator
                    try:
                        pe = atoms.get_potential_energy()
                    except:
                        print(calc_failure_msg_template.format(
                            structure_name, scale_suffix))
                        print(traceback.format_exc())
                        pe = np.nan

                    print("dist: {}, PE: {}".format(scale_suffix, pe))
                    data["dist"].append(scale_suffix)
                    data["PE"].append(pe)

            else:
                raise Exception(
                    "For an enetry of unit_structures, either 'filename_scale_suffixes' or 'interatomic_distances' field should be provide!!")

            df_mlff = pd.DataFrame(data, columns=['dist', "PE"])
            df_mlff.set_index("dist", inplace=True)

            out_dir = Path(self.config["res_out_dir"])
            os.makedirs(out_dir, exist_ok=True)
            structure_energy_save_name = f"{self.config['energy_save_name']}_{structure_name}.csv"
            df_mlff.to_csv(out_dir / structure_energy_save_name)

            df_ref = self.load_reference_results(
                unit_structure_dict["reference_scale_energy_relation_path"])

            structure_error_save_name = f"{self.config['error_save_name']}_{structure_name}.txt"
            self.calculate_pe_well_error_metric(
                df_ref,
                df_mlff,
                out_dir / structure_error_save_name
            )

            self.generate_comparison_plot(
                df_ref,
                df_mlff,
                out_dir / f"compare_pe_well_{structure_name}.png",
                save_res=True
            )
