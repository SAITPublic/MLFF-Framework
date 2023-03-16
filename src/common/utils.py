import os
import logging
import copy
import datetime
import time
import importlib
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

import torch

from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.registry import registry 

from src.common import distutils as benchmark_distutils
from src.common.config import check_config
from src.common.registry import evaluator_registry


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
def setup_benchmark_logging(config, file_log=True):
    root = logging.getLogger()
    bm_logging = logging.getLogger("BenchmarkLogging")
    if distutils.is_master():
        if not config["is_debug"]:
            # The initial logging setup is performed by setup_logging() of ocpmodels.common.utils at main.py.
            # We'll follow the logging format.
            log_formatter = root.handlers[0].formatter

            # setup for benchmark logging
            # inherit root logging and remove it
            for handler in root.handlers:
                bm_logging.addHandler(handler)
                root.removeHandler(handler)
            
            if file_log:
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
def setup_benchmark_imports(config: Optional[dict] = None):
    # First, check if imports are already setup
    has_already_setup = registry.get("imports_benchmark_setup", no_warning=True)
    if has_already_setup:
        return

    has_already_setup = evaluator_registry.get("imports_benchmark_setup", no_warning=True)
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
        for key in ["trainers", "datasets", "models", "tasks", "evaluators"]:
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
        evaluator_registry.register("imports_benchmark_setup", True)


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
   
    original_config = config
    config = copy.deepcopy(original_config)
    if args.distributed:
        benchmark_distutils.setup(config)
        if config["gp_gpus"] is not None:
            gp_utils.setup_gp(config)
    
    # make timestamp_id not empty
    config["timestamp_id"] = _set_timestamp_id(config) 

    # check whether arguments which are required to initiate a Trainer class exist in a configuration
    confing = check_config(config)

    start_time = time.time()
    try:
        # setup benchmark logging with a file handler
        setup_benchmark_logging(config, file_log=(config.get("logger", None) == "files"))
        setup_benchmark_imports(config)

        trainer_class = registry.get_trainer_class(config.get("trainer", "forces"))
        assert trainer_class is not None, "Trainer not found"
        trainer = trainer_class(config = config)

        task_cls = registry.get_task_class(config["mode"])
        assert task_cls is not None, "Task not found"
        task = task_cls(config)
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
   
    original_config = config
    config = copy.deepcopy(original_config)
    
    start_time = time.time()
    try:
        # setup benchmark logging with a file handler
        setup_benchmark_logging(config, file_log=False)
        setup_benchmark_imports(config)

        evaluator_class = evaluator_registry.get_evaluator_class(config.get("evaluator", "rbf"))
        assert evaluator_class is not None, "Evaluator not found"
        evaluator = evaluator_class(config = config)

        task_cls = registry.get_task_class(config["mode"])
        assert task_cls is not None, "Task not found"
        task = task_cls(config)
        ctx = _EvaluationContext(config=config, task=task, evaluator=evaluator)
        yield ctx
        distutils.synchronize()
    finally:
        total_time = time.time()-start_time
        bm_logging.info(f"Total time taken: {total_time:.1f} sec ({total_time/3600:.1f} h)")
        if args.distributed:
            distutils.cleanup()
