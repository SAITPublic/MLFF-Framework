"""
Copied from ocp.main.py
Modifications:
1) deal with additional benchmark arguments
"""

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(1, os.path.abspath("./codebases/ocp")) 
sys.path.insert(2, os.path.abspath("./codebases/nequip")) 
sys.path.insert(3, os.path.abspath("./codebases/allegro")) 
#sys.path.insert(4, os.path.abspath("./codebases/mace")) 

import copy
import logging
from pathlib import Path

import submitit

from ocpmodels.common.utils import (
    setup_logging,
    build_config,
    create_grid,
    save_experiment_log,
)

# benchmark codes
from src.common.flags import benchmark_flags
from src.common.config import (
    add_benchmark_config, 
    add_benchmark_fit_scale_config,
    add_benchmark_validate_config,
    add_benchmark_evaluate_config,
)
from src.common.utils import new_trainer_context, new_evaluator_context


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        if config["mode"] == "evaluate":
            with new_evaluator_context(args=args, config=config) as ctx:
                self.config = ctx.config
                self.task = ctx.task
                self.evaluator = ctx.evaluator

                self.task.setup(self.evaluator)
                self.task.run()
        else:
            # train, fit-scale, validate
            with new_trainer_context(args=args, config=config) as ctx:
                self.config = ctx.config
                self.task = ctx.task
                self.trainer = ctx.trainer

                self.task.setup(self.trainer)
                self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


if __name__ == "__main__":
    setup_logging()

    parser = benchmark_flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    config = add_benchmark_config(config, args)
    config = add_benchmark_fit_scale_config(config, args)
    config = add_benchmark_validate_config(config, args)
    config = add_benchmark_evaluate_config(config, args)

    if args.submit:  # Run on cluster
        slurm_add_params = config.get(
            "slurm", None
        )  # additional slurm arguments
        if args.sweep_yml:  # Run grid search
            configs = create_grid(config, args.sweep_yml)
        else:
            configs = [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = submitit.AutoExecutor(
            folder=args.logdir / "%j", slurm_max_num_timeout=3
        )
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(config["optim"]["num_workers"] + 1),
            tasks_per_node=(args.num_gpus if args.distributed else 1),
            nodes=args.num_nodes,
            slurm_additional_parameters=slurm_add_params,
        )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(Runner(), configs)
        logging.info(
            f"Submitted jobs: {', '.join([job.job_id for job in jobs])}"
        )
        log_file = save_experiment_log(args, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:  # Run locally
        Runner()(config)
