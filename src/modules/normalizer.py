"""
written by robert.cho and byunggook.na (SAIT)
"""

import torch
from ocpmodels.modules.normalizer import Normalizer

from src.common.utils import bm_logging

class NormalizerPerAtom(Normalizer):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        assert tensor is None, "tensor should be None when NormalizerPerAtom is used"
        super().__init__(mean=mean, std=std, device=device)

    def norm(self, tensor, N):
        return (tensor - self.mean * N) / self.std

    def denorm(self, normed_tensor, N):
        return normed_tensor * self.std + self.mean * N


def log_and_check_normalizers(energy_normalizer, forces_normalizer, loaded=False):
    if loaded:
        bm_logging.info(f"Set normalizers from the checkpoint")
    else:
        bm_logging.info(f"Set normalizers by the config")
    bm_logging.info(f" - energy uses {type(energy_normalizer).__name__} (shift: {energy_normalizer.mean}, scale: {energy_normalizer.std})")
    bm_logging.info(f" - forces uses {type(forces_normalizer).__name__} (shift: {forces_normalizer.mean}, scale: {forces_normalizer.std})")
    assert forces_normalizer.mean == 0, "mean of forces should be zero"
    assert forces_normalizer.std  == energy_normalizer.std, "scales of energy and forces should be identical"