import inspect
import math
import torch
from ocpmodels.common import distutils

"""
Available LR scheduler

0) ConstantLR
1) LinearLR
2) ReduceLROnPlateau
3) CosineAnnealingWarmRestarts
4) ExponentialLR
5) LambdaLR (or not specified in a configuration yaml file)

"""

def convert_epoch_to_step(epoch, config):
    bs = float(config["batch_size"]) * distutils.get_world_size()
    step = math.ceil(config["num_train"]/bs) * epoch
    return step


class LinearWarmupLRMultiplier:
    """
    Returns a learning rate multiplier.
    Till `warmup_steps`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by a multiplier calculated by the defined functions in this class every step.
    """    
    def __init__(self, config):
        self.config = config
        if "warmup_steps" in self.config.keys():
            self.warmup_steps = float(self.config["warmup_steps"])
        elif "warmup_epochs" in self.config.keys():
            self.warmup_steps = convert_epoch_to_step(self.config["warmup_epochs"], config)
        else:
            raise Exception("When using warmup, it requires 'warmup_steps' or 'warmup_epochs' in the config.")
        # lr at starting point, which is multiplied by warmup factor.
        # default value is (1 / warmup steps).
        self.warmup_factor = self.config.get("warmup_factor", 1.0/self.warmup_steps) 

    def _warmup_multiplier(self, current_step):
        alpha = current_step / self.warmup_steps
        return self.warmup_factor * (1.0 - alpha) + alpha

    def warmup_then_constant(self, current_step):
        if current_step <= self.warmup_steps:
            return self._warmup_multiplier(current_step)
        else:
            return 1

    def get_lr_lambda(self):
        return lambda x: self.warmup_then_constant(x)


class LinearWarmupStepDecayLRMultiplier(LinearWarmupLRMultiplier):
    def __init__(self, config):
        super().__init__(config)
        if "lr_milestone_steps" in self.config.keys():
            self.lr_milestone_steps = self.config["lr_milestone_steps"]
        elif "lr_milestone_epochs" in self.config.keys():
            self.lr_milestone_steps = [
                int(convert_epoch_to_step(ep, config)) for ep in self.config["lr_milestone_epochs"]
            ]
        else:
            raise NotImplementedError("When using step decaying with warmup, it requires 'lr_milestone_steps' or 'lr_milestone_epochs' in the config.")

    def warmup_then_step_decay(self, current_step):
        """After the warmup phase, 
        lr gets multiplied by `lr_gamma` every time a milestone is crossed.
        """
        if current_step <= self.warmup_steps:
            return self._warmup_multiplier(current_step)
        else:
            idx = bisect(self.lr_milestone_steps, current_step)
            return pow(self.config["lr_gamma"], idx)
    
    def get_lr_lambda(self):
        return lambda x: self.warmup_then_step_decay(x)


class LinearWarmupLinearDecayLRMultiplier(LinearWarmupLRMultiplier):
    def __init__(self, config):
        super().__init__(config)
        self.total_decay_steps = convert_epoch_to_step(self.config["max_epochs"], config) - self.warmup_steps

    def warmup_then_linear_decay(self, current_step):
        """After the warmup phase,
        lr decreases linearly.
        Default as in LinearLR (start_factor = 1.0, end_factor = 0.0)
        """
        if current_step <= self.warmup_steps:
            return self._warmup_multiplier(current_step)
        else:
            current_step -= self.warmup_steps
            return (self.total_decay_steps - current_step) / self.total_decay_steps

    def get_lr_lambda(self):
        return lambda x: self.warmup_then_linear_decay(x)


class LRScheduler:
    """
    Copied from ocp/ocpmodels/modules/scheduler.py
    Modifications:
    1) add different lambda functions to deal with various warmup-based decaying strategies
    """
    # scheduler_types = [
    #     "LinearLR", # SAIT
    #     "ReduceLROnPlateau", # OCP, NequIP, MACE
    #     "CosineAnnealingWarmRestarts", # NequIP
    #     "ExponentialLR", # MACE
    #     "LambdaLR", # OCP (for warmup)
    # ]

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config.copy()

        # if scheduler is not specified, default is LambdaLR
        self.scheduler_type = self.config.get("scheduler", "LambdaLR") 

        # add required hyper-parameters for each scheduler
        if self.scheduler_type == "ConstantLR":
            self.config["total_iters"] = convert_epoch_to_step(self.config["max_epochs"], self.config)
            self.config["factor"] = self.config.get("factor", 1.0)
        elif self.scheduler_type == "LinearLR":
            self.config["total_iters"] = convert_epoch_to_step(self.config["max_epochs"], self.config)
            self.config["start_factor"] = self.config.get("start_factor", 1.0) # default setting
            self.config["end_factor"] = self.config.get("end_factor", 0.0) # default setting
        elif self.scheduler_type == "LambdaLR":
            # To consider warmup, lr decaying functions are defined
            assert ("warmup_steps" in self.config) or ("warmup_epochs" in self.config)
            assert ("lr_lambda" in self.config)
            self.scheduler_type = "LambdaLR"
            lr_lambda = self.config["lr_lambda"]

            # lr decaying strategy
            # 1) constant
            # 2) step
            # 3) linear
            if lr_lambda == "constant":
                lr_multiplier = LinearWarmupLRMultiplier(self.config)
            elif lr_lambda == "step":
                lr_multiplier = LinearWarmupStepDecayLRMultiplier(self.config)
            elif lr_lambda == "linear":
                lr_multiplier = LinearWarmupLinearDecayLRMultiplier(self.config)
            self.config["lr_lambda"] = lr_multiplier.get_lr_lambda()
        
        # extract hyper-parameters to initiate a scheduler
        scheduler_class = getattr(torch.optim.lr_scheduler, self.scheduler_type)
        scheduler_args = self._filter_kwargs_from_config(scheduler_class)
        self.scheduler = scheduler_class(optimizer, **scheduler_args)

    def _filter_kwargs_from_config(self, scheduler_class):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(scheduler_class)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        filter_keys.remove("optimizer")
        scheduler_args = {
            arg: self.config[arg] for arg in self.config if arg in filter_keys
        }
        return scheduler_args

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception(
                    "Validation set required for ReduceLROnPlateau."
                )
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]
