"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import torch
from pathlib import Path

from ocpmodels.common.registry import registry

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper

from src.trainers.forces_trainer import ForcesTrainer
from src.common.collaters.parallel_collater_nequip import ParallelCollaterNequIP
from src.common.utils import bm_logging


@registry.register_trainer("forces_nequip")
class NequIPForcesTrainer(ForcesTrainer):
    """
    Trainer class for the S2EF (Structure to Energy & Force) task, 
    and this class is especially used to train NequIP or Allegro models.
    """
    def __init__(self, config):
        super().__init__(config)

        # copied from nequip/scripts/train.py
        if int(torch.__version__.split(".")[1]) >= 11:
            # PyTorch >= 1.11
            torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
        else:
            torch._C._jit_set_bailout_depth(2)
    
    def _parse_config(self, config):
        trainer_config = super()._parse_config(config)

        # NequIP and Allegro do not need OCP normalizer (they use own normaliation strategy)
        ocp_normalize_flag = False
        data_config_style = trainer_config.get("data_config_style", "OCP")
        if data_config_style == "OCP":
            ocp_normalize_flag = trainer_config["dataset"].get("normalize_labels", False)
            trainer_config["dataset"]["normalize_labels"] = False
        elif data_config_style == "SAIT":
            ocp_normalize_flag = trainer_config["normalizer"].get("normalize_labels", False)
            trainer_config["normalizer"]["normalize_labels"] = False
        if ocp_normalize_flag:
            bm_logging.info("In the given configuration file or the configuration saved in the checkpoint, `normalize_labels` is set as `True` ")
            bm_logging.info("  NequIP and Allegro do not need OCP normalizers, instead they use own normalization strategy by employing scale and shift layers.")
            bm_logging.info("  Hence `normalize_labels` will be changed as `False` to turn off the OCP normalizer operation.")
            bm_logging.info("  You can control their own normalization strategy by turning on/off `use_scale_shift` in the model configuration.")
        
        if "data_normalization" in trainer_config["model_attributes"].keys():
            bm_logging.info("(deprecated) The configuration saved in the checkpoint is old-styled. `data_normalization` is converted into `use_scale_shift`.")
            trainer_config["model_attributes"]["use_scale_shift"] = trainer_config["model_attributes"]["data_normalization"]
            del trainer_config["model_attributes"]["data_normalization"]

        if (trainer_config["model_attributes"].get("use_scale_shift", True) 
            or trainer_config["model_attributes"].get("avg_num_neighbors", None) == "auto"
        ):
            if data_config_style == "OCP":
                # OCP data config style
                trainer_config["model_attributes"]["dataset"] = trainer_config["dataset"]["src"]
            elif data_config_style == "SAIT":
                # SAIT data config style
                assert isinstance(trainer_config["dataset"], list)
                if len(trainer_config["dataset"]) > 1:
                    bm_logging("The first source of training datasets will be used to obtain avg_num_neighbors, scale, or shift.")
                if not Path(trainer_config["dataset"][0]["src"]).exists():
                    raise RuntimeError("The lmdb source file or directory should be specified.")
                trainer_config["model_attributes"]["dataset"] = trainer_config["dataset"][0]["src"]

        if self.mode == "validate":
            trainer_config["model_attributes"]["initialize"] = False
        return trainer_config
    
    def initiate_collater(self):
        self.type_mapper = TypeMapper(
            type_names=self.config["model_attributes"].get("type_names", None),
            chemical_symbol_to_type=self.config["model_attributes"].get("chemical_symbol_to_type", None),
            chemical_symbols=self.config["model_attributes"].get("chemical_symbols", None),
        )
        return ParallelCollaterNequIP(
            num_gpus=0 if self.cpu else 1,
            otf_graph=self.config["model_attributes"].get("otf_graph", False),
            use_pbc=self.config["model_attributes"].get("use_pbc", False),
            type_mapper=self.type_mapper,
        )

    def _split_trainable_params_optimizer_weight_decay(self):
        # as in nequip code
        # there is no splitting params according to weight decaying
        params_decay = self.model.parameters()
        params_no_decay = []
        return params_decay, params_no_decay

    def _compute_loss(self, out, batch_list):
        # loss function always needs to be in normalized unit (according to NequIP)
        if self.model.training:
            # loss used in train mode
            # target is converted real unit -> normalized unit 
            # (prediction is not touched, because it is in normalized unit)
            normalized_batch_list = []
            for batch in batch_list:
                b = self._unwrapped_model.do_unscale(
                    data=AtomicData.to_AtomicDataDict(batch)
                )
                normalized_batch = AtomicData.from_AtomicDataDict(b)
                normalized_batch.y = b[AtomicDataDict.TOTAL_ENERGY_KEY]
                normalized_batch.force = b[AtomicDataDict.FORCE_KEY]
                if self.use_stress:
                    normalized_batch.stress = b[AtomicDataDict.STRESS_KEY]
                normalized_batch.natoms = torch.bincount(batch[AtomicDataDict.BATCH_KEY])
                normalized_batch_list.append(normalized_batch)
            return super()._compute_loss(out=out, batch_list=normalized_batch_list)
        else:
            # loss used in eval mode
            with torch.no_grad():
                # target is converted real unit -> normalized unit
                normalized_batch_list = []
                for batch in batch_list:
                    b = self._unwrapped_model.do_unscale(
                        data=AtomicData.to_AtomicDataDict(batch),
                        force_process=True,
                    )
                    normalized_batch = AtomicData.from_AtomicDataDict(b)
                    normalized_batch.y = b[AtomicDataDict.TOTAL_ENERGY_KEY]
                    normalized_batch.force = b[AtomicDataDict.FORCE_KEY]
                    if self.use_stress:
                        normalized_batch.stress = b[AtomicDataDict.STRESS_KEY]
                    normalized_batch.natoms = torch.bincount(batch[AtomicDataDict.BATCH_KEY])
                    normalized_batch_list.append(normalized_batch)
                # prediction is converted ? -> normalized unit
                nequip_ocp_key_mapper = [
                    [AtomicDataDict.TOTAL_ENERGY_KEY, "energy"],
                    [AtomicDataDict.FORCE_KEY, "forces"],
                ]
                if self.use_stress:
                    nequip_ocp_key_mapper.append([AtomicDataDict.STRESS_KEY, "stress"])

                _out = self._unwrapped_model.do_unscale(
                    data={key_map[0]: out[key_map[1]] for key_map in nequip_ocp_key_mapper},
                    force_process=True,
                )
                normalized_out = {key_map[1]: _out[key_map[0]] for key_map in nequip_ocp_key_mapper}
                return super()._compute_loss(out=normalized_out, batch_list=normalized_batch_list)

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        
        for batch in batch_list:
            batch.y = batch[AtomicDataDict.TOTAL_ENERGY_KEY]
            batch.force = batch[AtomicDataDict.FORCE_KEY]
            batch.natoms = torch.bincount(batch[AtomicDataDict.BATCH_KEY])

        if self.model.training:
            # train mode
            # prediction is converted normalized unit -> real unit
            with torch.no_grad():
                _unscaled = {
                    AtomicDataDict.TOTAL_ENERGY_KEY: out["energy"], 
                    AtomicDataDict.FORCE_KEY: out["forces"],
                }
                if self.use_stress:
                    _unscaled[AtomicDataDict.STRESS_KEY] = out["stress"]
                _out = self._unwrapped_model.do_scale(
                    _unscaled,
                    force_process=True,
                )
                out["energy"] = _out[AtomicDataDict.TOTAL_ENERGY_KEY]
                out["forces"] = _out[AtomicDataDict.FORCE_KEY]
                if self.use_stress:
                    out["stress"] = _out[AtomicDataDict.STRESS_KEY]
        
        return super()._compute_metrics(out=out, batch_list=batch_list, evaluator=evaluator, metrics=metrics)

    def make_checkpoint_dict(self, metrics, training_state):
        if "dataset" in self.config["model_attributes"]:
            del self.config["model_attributes"]["dataset"]
        
        self.config["model_attributes"]["avg_num_neighbors"] = self._unwrapped_model.avg_num_neighbors

        ckpt_dict = super().make_checkpoint_dict(metrics, training_state)
        return ckpt_dict