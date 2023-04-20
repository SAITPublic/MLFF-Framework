import os
import logging
import copy
import datetime
import time
import importlib
import yaml
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path
import numpy as np
import torch
from math import log10, floor, isnan

from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.utils import load_config
from ocpmodels.common.registry import registry 

from src.common import distutils as benchmark_distutils
from src.common.config import check_config
from src.common.registry import md_evaluate_registry


def get_device(config):
    assert "local_rank" in config
    if torch.cuda.is_available() and (config.get("gpus", 1) > 0):
        return torch.device(f"cuda:{config['local_rank']}")
    else:
        return torch.device("cpu")


# In benchmark, we mainly use the logger named bm_logging
bm_logging = logging.getLogger("BenchmarkLogging")


# benchmark logger setting considering distributed environment 
# (we can omit 'if is_master()' when using logging)
def setup_benchmark_logging(config):
    root = logging.getLogger()
    bm_logging = logging.getLogger("BenchmarkLogging")
    if distutils.is_master():
        # if not config.get("is_debug", False) and config["mode"] == "train":
        if True:
            # The initial logging setup is performed by setup_logging() of ocpmodels.common.utils at main.py.
            # We'll follow the logging format.
            log_formatter = root.handlers[0].formatter

            # setup for benchmark logging
            # inherit root logging and remove it
            for handler in root.handlers:
                bm_logging.addHandler(handler)
                root.removeHandler(handler)
            
            if config.get("logger", None) == "files" and config["mode"] == "train":
                # send INFO to a file
                logger_name = config["logger"] if isinstance(config["logger"], str) else config["logger"]["name"]
                logdir = os.path.join(config["run_dir"], "logs", logger_name, config["timestamp_id"])
                os.makedirs(logdir, exist_ok=True)
                log_path = os.path.join(logdir, "experiment.log")
                file_handler = logging.FileHandler(log_path)
                file_handler.setFormatter(log_formatter)
                bm_logging.addHandler(file_handler)
    else:
        # disable logging by other ranks
        for handler in root.handlers:
            root.removeHandler(handler)


# Copied from ocp.ocpmodels.utils
def setup_benchmark_imports(config=None):
    # First, check if imports are already setup
    has_already_setup = registry.get("imports_benchmark_setup", no_warning=True)
    if has_already_setup:
        return

    has_already_setup = md_evaluate_registry.get("imports_benchmark_setup", no_warning=True)
    if has_already_setup:
        return

    try:
        this_utils_filepath = Path(__file__).resolve().absolute()
        benchmark_root = this_utils_filepath.parent.parent.parent
        logging.info(f"Project root: {benchmark_root}")

        # OCP
        importlib.import_module("ocpmodels.common.logger")
        for key in ["trainers", "datasets", "models", "tasks"]:
            for path in (benchmark_root / "codebases" / "ocp" / "ocpmodels" / key).rglob("*.py"):
                module_name = ".".join(
                    path.absolute()
                    .relative_to(benchmark_root.absolute())
                    .with_suffix("")
                    .parts
                )
                importlib.import_module(module_name)

        # SAIT-MLFF-Framework
        # : re-define classes of trainers and tasks
        importlib.import_module("src.common.logger")
        for key in ["trainers", "datasets", "models", "tasks", "md_evaluate"]:
            for path in (benchmark_root / "src" / key).rglob("*.py"):
                module_name = ".".join(
                    path.absolute()
                    .relative_to(benchmark_root.absolute())
                    .with_suffix("")
                    .parts
                )
                importlib.import_module(module_name)
    finally:
        registry.register("imports_benchmark_setup", True)
        md_evaluate_registry.register("imports_benchmark_setup", True)


@contextmanager
def new_trainer_context(*, config: Dict[str, Any], args: Namespace):
    """
    Copied from ocp/ocpmodels/common/utils.py
    Modifications:
    1) specify timestamp_id
    2) use benchmark_distutils.setup() instead of distutils.setup()
    """
    @dataclass
    class _TrainingContext:
        config: Dict[str, Any]
        task: "BaseTask"
        trainer: "BaseTrainer"

    def _set_timestamp_id(config):
        if config["timestamp_id"] is None:
            # merging timestamp and expr ID when timestamp_id is empty
            timestamp = torch.tensor(int(datetime.datetime.now().timestamp())).to(get_device(config))
            distutils.broadcast(timestamp, 0)
            timestamp = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            if config["identifier"] in ["", None]:
                return timestamp
            else:
                return f"{config['identifier']}-{timestamp}"
   
    original_config = copy.deepcopy(config)
    if args.distributed:
        benchmark_distutils.setup(config)
        if config["gp_gpus"] is not None:
            gp_utils.setup_gp(config)
    
    # make timestamp_id not empty
    config["timestamp_id"] = _set_timestamp_id(config) 

    # check whether arguments which are required to initiate a Trainer class exist in a configuration
    config = check_config(config)

    start_time = time.time()
    try:
        # setup benchmark logging with a file handler
        setup_benchmark_logging(config)
        setup_benchmark_imports(config)

        # construct a trainer instance
        trainer_class = registry.get_trainer_class(config.get("trainer", "forces"))
        assert trainer_class is not None, "Trainer class is not found"
        trainer = trainer_class(config = config)

        if config["mode"] == "train":
            # save a training configuration yaml file into checkpoint_dir
            with open(os.path.join(trainer.config["cmd"]["checkpoint_dir"], "config_train.yml"), 'w') as f:
                input_config, _, _ = load_config(args.config_yml)
                yaml.dump(input_config, f)

        # construct a task instance (given a trainer)
        task_cls = registry.get_task_class(config["mode"])
        assert task_cls is not None, "Task is not found"
        task = task_cls(config=original_config)
        ctx = _TrainingContext(config=original_config, task=task, trainer=trainer)
        yield ctx
        distutils.synchronize()
    finally:
        total_time = time.time()-start_time
        bm_logging.info(f"Total time taken: {total_time:.1f} sec ({total_time/3600:.1f} h)")
        if args.distributed:
            distutils.cleanup()


@contextmanager
def new_evaluator_context(*, config: Dict[str, Any], args: Namespace):
    @dataclass
    class _EvaluationContext:
        config: Dict[str, Any]
        task: "BaseTask"
        evaluator: "BaseEvaluator"
   
    original_config = copy.deepcopy(config)
    start_time = time.time()
    try:
        # setup benchmark logging with a file handler
        setup_benchmark_logging(config)
        setup_benchmark_imports(config)

        # construct an evaluator or a simulator
        if config["mode"] == "run-md":
            evaluator_class = md_evaluate_registry.get_md_evaluate_class("simulator")
        else:
            evaluator_class = md_evaluate_registry.get_md_evaluate_class(config["evaluation_metric"])
            assert evaluator_class is not None, f"Evaluator class is not found"
        evaluator = evaluator_class(config = config)

        # construct a task instance
        task_cls = registry.get_task_class(config["mode"])
        assert task_cls is not None, "Task class is not found"
        task = task_cls(config=original_config)
        ctx = _EvaluationContext(config=original_config, task=task, evaluator=evaluator)
        yield ctx
    finally:
        total_time = time.time()-start_time
        bm_logging.info(f"Total time taken: {total_time:.1f} sec ({total_time/3600:.1f} h)")
        
        
def calc_error_metric(f_predict, f_target, metric_name):
    metric_name_lower = metric_name.lower()
    if metric_name_lower == "mae":
        return np.mean(np.absolute(f_predict - f_target))
    elif metric_name_lower == "rmse":
        return np.sqrt(np.mean((f_predict - f_target)**2))
    else:
        raise Exception("Provided metric name '{}' is not supported!".format(metric_name))
    

def truncate_float(x, n_significant):
    return round(x, (n_significant-1) - int(floor(log10(abs(x)))))
