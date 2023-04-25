"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from ocpmodels.modules.normalizer import Normalizer

class NormalizerPerAtom(Normalizer):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        super().__init__(tensor,mean,std,device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def norm(self, tensor,N):
        return (tensor - self.mean*N) / self.std

    def denorm(self, normed_tensor,N):
        return normed_tensor * self.std + self.mean*N

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)
