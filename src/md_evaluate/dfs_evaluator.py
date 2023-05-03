"""
Written by byunggook.na and heesun88.lee
"""
from pathlib import Path
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from ase import io
from ase.build.supercells import make_supercell
from FOX import MultiMolecule

from src.common.utils import calc_error_metric
from src.common.registry import md_evaluate_registry
from src.md_evaluate.base_evaluator import BaseEvaluator

@md_evaluate_registry.register_md_evaluate("df")
@md_evaluate_registry.register_md_evaluate("distribution-functions")
class DFEvaluator(BaseEvaluator):

    def calculate_rdf_fox(self, traj_atoms, out_identifier):
        mols = MultiMolecule.from_ase(traj_atoms)
        # assume periodic in all axis as RDF is meaningful only in periodic system...
        rdf = mols.init_rdf(
            periodic="xyz", dr=self.config["dr_rdf"], r_max=self.config["r_max_rdf"])
        pair_list = rdf.columns.values.tolist()

        filename_rdf = "RDF_{}.csv".format(out_identifier)
        rdf.to_csv(Path(self.config["res_out_dir"]) / filename_rdf)

        return rdf, pair_list
    
    def calculate_adf_fox(self, traj_atoms, out_identifier):
        mols = MultiMolecule.from_ase(traj_atoms)
        # assume periodic in all axis as ADF is meaningful only in periodic system...
        adf = mols.init_adf(
            periodic="xyz", r_max=self.config["r_max_adf"])  # , weight=None
        triplet_list = adf.columns.values.tolist()

        filename_adf = "ADF_{}.csv".format(out_identifier)
        adf.to_csv(Path(self.config["res_out_dir"]) / filename_adf)

        return adf, triplet_list

    def generate_comparison_figure(self, distribution_ref, distribution_dict_mlff, combination_list, fig_name):
        x_axis_name = distribution_ref.index.name
        label_list = ["AI_MD"] + list(distribution_dict_mlff.keys())
        for combination in combination_list:
            plt.figure()
            ax = distribution_ref.plot(y=combination, use_index=True)
            for df_mlff in distribution_dict_mlff.values():
                df_mlff.plot(y=combination, use_index=True, ax=ax)

            ax.set_xlabel(x_axis_name)
            ax.set_ylabel('Distrubution Func.')
            ax.set_title(combination)
            ax.legend(label_list)
            plt.savefig(Path(self.config["res_out_dir"]) /
                        '{}_{}.png'.format(fig_name, combination))
            # plt.show()
            # # plt.close('all')
    
    def output_error_metrics(self, distribution_ref, distribution_dict_mlff, combination_list, file_name):
        distribution_error_dict = defaultdict(dict)
        for mlff_uid, distribution_df in distribution_dict_mlff.items():
            for combination in combination_list:
                distribution_error_dict[mlff_uid][combination] = \
                    calc_error_metric(distribution_df[combination].values,
                                      distribution_ref[combination].values,
                                      'mae')

            distribution_error_dict[mlff_uid]['average'] = \
                sum(distribution_error_dict[mlff_uid].values()) \
                    / len(distribution_error_dict[mlff_uid])

        out_file_path = Path(
            self.config["res_out_dir"]) / "{}.dat".format(file_name)
        with open(out_file_path, 'w') as f:
            json.dump(distribution_error_dict, f, indent=4)

    @staticmethod
    def get_traj_atoms(path, index, format, n_extend):
        traj_atoms = io.read(path, index=index, format=format)

        if n_extend is not None and n_extend != 1:
            traj_atoms_extended = []
            for atoms in traj_atoms:
                atoms.wrap()
                atoms_extended = make_supercell(
                    atoms, [[n_extend, 0, 0], [0, n_extend, 0], [0, 0, n_extend]], wrap=True)
                traj_atoms_extended.append(atoms_extended)
            traj_atoms = traj_atoms_extended

        return traj_atoms

    def evaluate(self):
        Path(self.config["res_out_dir"]).mkdir(parents=True, exist_ok=True)

        trajs_atoms_ai = DFEvaluator.get_traj_atoms(
            self.config["ai_md_traj"]["path"],
            index=self.config["ai_md_traj"].get("index", ":"),
            format=self.config["ai_md_traj"].get("format"),
            n_extend=self.config["ai_md_traj"].get("n_extend"))
        out_identifier_ai = "AIMD_".format(
            self.config["ai_md_traj"]["out_identifier"]
        )

        rdf_ref, pair_list_ref = self.calculate_rdf_fox(
            trajs_atoms_ai, out_identifier_ai)
        adf_ref, triplet_list_ref = self.calculate_adf_fox(
            trajs_atoms_ai, out_identifier_ai)

        rdf_dict_mlff = {}
        adf_dict_mlff = {}
        for mlff_uid, mlff_traj_dict in self.config["mlff_md_traj"].items():
            traj_atoms_mlff = \
                DFEvaluator.get_traj_atoms(
                    mlff_traj_dict["path"],
                    index=mlff_traj_dict.get("index", ":"),
                    format=mlff_traj_dict.get("format"),
                    n_extend=mlff_traj_dict.get("n_extend")
                )
            out_identifier_mlff = "{}_{}".format(mlff_uid,
                                                 mlff_traj_dict["out_identifier"])

            self.logger.info(
                "Start calculating distrubution functions for model '{}'".format(mlff_uid))
            rdf_dict_mlff[mlff_uid], pair_list_mlff = \
                self.calculate_rdf_fox(traj_atoms_mlff, out_identifier_mlff)
            assert pair_list_ref == pair_list_mlff, \
                "Atom types in 'mlff_md_traj' should be the same as those in 'ai_md_traj'"

            adf_dict_mlff[mlff_uid], _ = self.calculate_adf_fox(
                traj_atoms_mlff, out_identifier_mlff)

        self.generate_comparison_figure(rdf_ref, rdf_dict_mlff, pair_list_ref,
                                        fig_name='RDF_compare')
        self.generate_comparison_figure(adf_ref, adf_dict_mlff, triplet_list_ref,
                                        fig_name='ADF_compare')

        self.output_error_metrics(rdf_ref, rdf_dict_mlff, pair_list_ref,
                                  file_name='RDF_error')
        self.output_error_metrics(adf_ref, adf_dict_mlff, triplet_list_ref,
                                  file_name='ADF_error')
