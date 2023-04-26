"""
written by robert.cho and byunggook.na (SAIT)
"""

import torch
from ocpmodels.modules.normalizer import Normalizer

class NormalizerPerAtom(Normalizer):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        assert tensor is not None, "tensor should be None when NormalizerPerAtom is used"
        super().__init__(tensor,mean,std,device)


    def norm(self, tensor,N):
        return (tensor - self.mean*N) / self.std

    def denorm(self, normed_tensor,N):
        return normed_tensor * self.std + self.mean*N
