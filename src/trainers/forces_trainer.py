"""
Copied from ocp.ocpmodels.trainers.forces_trainer.py
Modifications:
1) modify class explanation
2) use a benchmark logger (named bm_logging) instead of the root logger (named logging)
"""

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import pathlib
import json 
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm
from prettytable import PrettyTable

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.util import ensure_fitted

# for modifications
from src.common.utils import bm_logging # benchmark logging
from src.common.logger import parse_logs
from src.trainers.base_trainer import BaseTrainer
from src.modules.normalizer import NormalizerPerAtom, log_and_check_normalizers


@registry.register_trainer("forces")
class ForcesTrainer(BaseTrainer):
    """
    Trainer class for the Structure to Energy & Force (S2EF) task.
    """
    def _set_normalizer(self):
        self.normalizers = {}
        if self.normalizer.get("normalize_labels", False):
            if self.mode in ["validate", "fit-scale"]:
                # just empty normalizer (which will be loaded from the given checkpoint)
                if self.normalizer.get("per_atom", False):
                    self.normalizers["target"] = NormalizerPerAtom(mean=0.0, std=1.0, device=self.device,)
                else:
                    self.normalizers["target"] = Normalizer(mean=0.0, std=1.0, device=self.device,)
                self.normalizers["grad_target"] = Normalizer(mean=0.0, std=1.0, device=self.device)
                if self.mode == "fit-scale":
                    bm_logging.info(f"Normalizers are not set")
                return

            # force normalizer
            if "grad_target_std" in self.normalizer:
                # Load precomputed mean and std of training set labels (specified in a configuration file)
                if "grad_target_mean" in self.normalizer:
                    bm_logging.info("`grad_target_mean` is ignored and set as 0 explicitly.")
                scale = self.normalizer["grad_target_std"]
            elif "normalize_labels_json" in self.normalizer:
                # Load precomputed mean and std of training set labels (specified in a json file outside from a configuration file)
                normalize_stats = json.load(open(self.normalizer["normalize_labels_json"], 'r'))
                if "force_mean":
                    bm_logging.info("`force_mean` is ignored and set as 0 explicitly.")
                scale = normalize_stats.get("force_std", 1.0)
            else:
                # Compute mean and std of training set labels.
                # : force is already tensor (which can have different shapes)
                forces_train = torch.concat([data.force for data in self.train_loader.dataset])
                scale = torch.std(forces_train), # 3-dim vetors -> scala value
            self.normalizers["grad_target"] = Normalizer(mean=0.0, std=scale, device=self.device)

            # energy normalizer
            if "target_mean" in self.normalizer:
                shift = self.normalizer["target_mean"]
                scale = self.normalizer.get("target_std", self.normalizers["grad_target"])
                if scale != self.normalizers["grad_target"].std:
                    bm_logging.warning(f"Scaling factors of energy and force are recommended to be equal")
            elif "normalize_labels_json" in self.normalizer:
                if self.normalizer.get("per_atom", False):
                    # per-atom energy mean
                    shift = normalize_stats["energy_per_atom_mean"]
                else:
                    shift = normalize_stats["energy_mean"]
                if "energy_std":
                    bm_logging.info("`energy_std` is ignored and set as the value of `force_std` explicitly.")
                scale = self.normalizers["grad_target"].std # energy scale factor should be force std
            else:
                
                if self.normalizer.get("per_atom", False):
                    # per-atom energy mean
                    energy_per_atom_train = torch.tensor([data.y / data.force.shape[0] for data in self.train_loader.dataset])
                    shift = torch.mean(energy_per_atom_train)
                else:
                    # total energy mean
                    energy_train = torch.tensor([data.y for data in self.train_loader.dataset])
                    shift = torch.mean(energy_train)
                scale = self.normalizers["grad_target"].std # energy scale factor should be force std

            if self.normalizer.get("per_atom", False):
                # per-atom energy 
                self.normalizers["target"] = NormalizerPerAtom(mean=shift, std=scale, device=self.device)
            else:
                self.normalizers["target"] = Normalizer(mean=shift, std=scale, device=self.device)

            # logging the status of normalizers
            log_and_check_normalizers(self.normalizers["target"], self.normalizers["grad_target"], loaded=False)
            
    def _set_task(self):
        # most models have a scaler energy output (meaning that num_targets = 1)
        self.num_targets = 1

        # this benchmark focuses on s2ef task, so regress_forces should be true
        if "regress_forces" in self.config["model_attributes"]:
            assert self.config["model_attributes"]["regress_forces"], "Correct `regress_forces` to be true"
        else:
            self.config["model_attributes"]["regress_forces"] = True

    def update_best(self, primary_metric, val_metrics):
        curr_metric = val_metrics[primary_metric]["metric"]
        if (("mae" in primary_metric and curr_metric < self.best_val_metric) or
            ("mae" not in primary_metric and curr_metric > self.best_val_metric)
        ):
            self.best_val_metric = curr_metric
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )

    def train(self):
        start_train_time = time.time()
        
        ensure_fitted(self._unwrapped_model, warn=True)
        if self.logger:
            self.logger.log_model_training_info(self._unwrapped_model)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.task_name]
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}
        
        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)
        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            start_epoch_time = time.time()
            self.train_sampler.set_epoch(epoch_int) # shuffle
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()
                
                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )

                # update local metrics (which will be aggregated across all ranks at print_every steps)
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                if (self.step % self.config["cmd"]["print_every"] == 0 or
                    self.step % len(self.train_loader) == 0
                ):
                    # 1) aggregate training results so far
                    # 2) print logging
                    # 3) reset metrics
                    aggregated_metrics = self.evaluator.aggregate(self.metrics)

                    log_dict = {k: aggregated_metrics[k]["metric"] for k in aggregated_metrics}
                    log_dict.update(
                        {
                            "lr": self.scheduler.get_lr(),
                            "epoch": self.epoch,
                            "step": self.step,
                        }
                    )
                    # stdout logging
                    bm_logging.info("[train] " + parse_logs(log_dict))

                    # logger logging
                    if self.logger:
                        self.logger.log(log_dict, step=self.step, split="train")

                    # reset metrics after logging
                    self.metrics = {}

                if (checkpoint_every != -1 and self.step % checkpoint_every == 0):
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(split="val")
                        self.update_best(primary_metric, val_metrics)

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(metrics=val_metrics[primary_metric]["metric"])
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)
        
            if (self.config["save_ckpt_every_epoch"] and 
                (epoch_int+1) % self.config["save_ckpt_every_epoch"] == 0
            ):
                # evaluation checkpoint (for benchmarking models during training)
                self.save(
                    metrics=val_metrics,
                    checkpoint_file=f"ckpt_ep{epoch_int+1}.pt",
                    training_state=False,
                )

            bm_logging.info(f"{epoch_int+1} epoch elapsed time: {time.time()-start_epoch_time:.1f} sec")

        train_elapsed_time = time.time()-start_train_time
        bm_logging.info(f"train() elapsed time: {train_elapsed_time:.1f} sec")

        # final evaluation
        bm_logging.info("Performing the final evaluation (last model)")
        metric_table = self.create_metric_table(display_meV=True)
        bm_logging.info(f"\n{metric_table}")
        if self.logger:
            self.logger.log_final_metrics(metric_table, train_elapsed_time)

        # end procedure of train()
        self._end_train()

    def _forward(self, batch_list):
        out_energy, out_forces = self.model(batch_list)
        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)
        out = {
            "energy": out_energy,
            "forces": out_forces,
        }
        return out

    def _compute_loss(self, out, batch_list):
        loss = []

        # set a mask to filter out fixed atoms
        if self.config["task"].get("train_on_free_atoms", False):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list], dim=0
            )
            free_mask = fixed == 0
            
        # Energy loss
        energy_mult = self.config["optim"].get("energy_coefficient", 1)

        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            if self.normalizer.get("per_atom", False):
                energy_target = self.normalizers["target"].norm(energy_target,batch_list[0].natoms)
            else:
                energy_target = self.normalizers["target"].norm(energy_target)

        if "per_atom" in self.config["optim"].get("loss_energy", "energy_per_atom_mse"):
            natoms = torch.cat(
                [batch.natoms.to(self.device) for batch in batch_list], dim=0
            )
            if self.config["task"].get("train_on_free_atoms", False):
                s_idx = 0
                natoms_free = []
                for n_at in natoms:
                    natoms_free.append(torch.sum(free_mask[s_idx : s_idx + n_at]).item())
                    s_idx += n_at
                natoms = torch.LongTensor(natoms_free).to(self.device)
            energy_loss = self.loss_fn["energy"](
                input=out["energy"], 
                target=energy_target, 
                natoms=natoms,
                batch_size=batch_list[0].natoms.shape[0],
            )
        else:
            energy_loss = self.loss_fn["energy"](
                input=out["energy"], 
                target=energy_target,
            )
        loss.append(energy_mult * energy_loss)

        # Force loss
        force_mult = self.config["optim"].get("force_coefficient", 30)

        force_target = torch.cat(
            [batch.force.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            force_target = self.normalizers["grad_target"].norm(force_target)        

        if self.config["task"].get("train_on_free_atoms", False):
            if (self.config["optim"].get("loss_force", "mse").startswith("atomwise")):
                force_mult = self.config["optim"].get("force_coefficient", 1)
                natoms = torch.cat(
                    [batch.natoms.to(self.device) for batch in batch_list], dim=0
                )
                natoms = torch.repeat_interleave(natoms, natoms)
                force_loss = self.loss_fn["force"](
                    input=out["forces"][free_mask],
                    target=force_target[free_mask],
                    natoms=natoms[free_mask],
                    batch_size=batch_list[0].natoms.shape[0],
                )
            else:
                force_loss = self.loss_fn["force"](
                    input=out["forces"][free_mask], 
                    target=force_target[free_mask],
                )
        else:
            force_loss = self.loss_fn["force"](
                input=out["forces"], 
                target=force_target,
            )
        # When force per dim loss is used, DDPLoss deals with "force loss / 3"
        loss.append(force_mult * force_loss)

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        target = {
            "energy": torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            ),
            "forces": torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": torch.cat(
                [batch.natoms.to(self.device) for batch in batch_list], dim=0
            ),
        }

        if self.config["task"].get("eval_on_free_atoms", True):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list]
            )
            mask_free = fixed == 0
            out["forces"] = out["forces"][mask_free]
            target["forces"] = target["forces"][mask_free]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(torch.sum(mask_free[s_idx : s_idx + natoms]).item())
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
        
        out["natoms"] = target["natoms"]

        # To calculate metrics, model output values are in real units
        if self.normalizer.get("normalize_labels", False):
            if self.normalizer.get("per_atom", False):
                N=torch.cat(
                [batch.natoms.to(self.device) for batch in batch_list], dim=0
                )
                out["energy"] = self.normalizers["target"].denorm(out["energy"],N)
            else:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics
