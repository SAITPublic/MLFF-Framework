import torch

from ocpmodels.common import distutils
from ocpmodels.modules.evaluator import (
    Evaluator,
    energy_mae,
    energy_mse,
    forcesx_mae,
    forcesx_mse,
    forcesy_mae,
    forcesy_mse,
    forcesz_mae,
    forcesz_mse,
    forces_mae,
    forces_mse,
    forces_cos,
    forces_magnitude,
    positions_mae,
    positions_mse,
    energy_force_within_threshold,
    energy_within_threshold,
    average_distance_within_threshold,
    min_diff,
    cosine_similarity,
    absolute_error,
    squared_error,
    magnitude_error,
)

"""
Available metrics:

Force:
    forcesx_mae
    forcesy_mae
    forcesz_mae
    forces_mae
    forces_cos
    forces_magnitude

Energy:
    energy_mae
    energy_mse
    energy_per_atom_mae
    energy_per_atom_mse
    energy_within_threshold
    energy_force_within_threshold

Position:
    average_distance_within_threshold
    positions_mae
    positions_mse


This benchmark focuses on s2ef task to train MLFF calculator which can be used in MD simulation.
Among the available metrics, we recommend metrics related force and energy.

Example (for brevity):
    forces_mae
    energy_mae_per_atom

"""

class MetricEvaluator(Evaluator):

    def __init__(self, task=None, task_metrics=None, task_attributes=None, task_primary_metric=None, device="cpu"):
        # this benchmark focuses on s2ef task
        assert task == "s2ef"
        super().__init__(task)

        # if users want new metrics and corresponding information, re-set the evaluator
        if task_metrics:
            self.task_metrics[task] = task_metrics
            self.metric_fn = self.task_metrics[task]
        if task_attributes:
            self.task_attributes[task] = task_attributes
        if task_primary_metric:
            assert task_primary_metric in self.task_metrics[task]
            self.task_primary_metric[task] = task_primary_metric

        self.device = device
    
    def eval(self, prediction, target, prev_metrics={}):
        for attr in self.task_attributes[self.task]:
            assert attr in prediction
            assert attr in target
            assert prediction[attr].shape == target[attr].shape

        metrics = prev_metrics
        for fn in self.task_metrics[self.task]:
            res = eval(fn)(prediction, target)
            metrics = self.update(fn, res, metrics)
        return metrics

    def aggregate(self, metrics):
        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        return aggregated_metrics


def energy_per_atom_mae(prediction, target):
    return absolute_error_per_atom(prediction["energy"], target["energy"], target["natoms"])


def energy_per_atom_mse(prediction, target):
    return squared_error_per_atom(prediction["energy"], target["energy"], target["natoms"])


def absolute_error_per_atom(prediction, target, natoms):
    error = torch.abs( (target - prediction) / natoms )
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": prediction.numel(),
    }


def squared_error_per_atom(prediction, target, natoms):
    error = ( (target - prediction) / natoms ) ** 2
    return {
        "metric": torch.mean(error).item(),
        "total": torch.sum(error).item(),
        "numel": prediction.numel(),
    }
