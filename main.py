"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

"""
Reference : ocp/ocpmodels/main.py

The following items are modified and they can be claimed as properties of Samsung Electronics. 

(1) Support more MLFF models (BPNN, NequIP, Allegro, and MACE)
(2) Support simulation indicators for the benchmark evaluation on simulations (RDF, ADF, EoS, PEW)
(3) Support more loss functions and metrics (loss.py and metric_evaluator.py in src/modules/)
(4) Support more learning rate schedulers (scheduler.py in src/modules/)
(5) Support normalization of per-atom energy (NormalizerPerAtom in src/modules/normalizer.py)
(6) Some different featurs are as follows:
    (a) Print training results using PrettyTable
    (b) Use a benchmark logger (named bm_logging) instead of the root logger (named logging in OCP)
    (c) Remove features that includes to save prediction results and make the corresponding directory named 'results'
    (d) Remove features related to HPO
    (e) Set the identifier of an experiment using the starting time
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
sys.path.insert(4, os.path.abspath("./codebases/mace")) 

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
    build_run_md_config,
    build_evaluate_config
)
from src.common.utils import new_trainer_context, new_evaluator_context


class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        if config["mode"] in ["train", "validate", "fit-scale"]:
            with new_trainer_context(args=args, config=config) as ctx:
                self.config = ctx.config
                self.trainer = ctx.trainer
                self.task = ctx.task
                self.task.setup(self.trainer)
                self.task.run()
        elif config["mode"] in ["run-md", "evaluate"]:
            with new_evaluator_context(args=args, config=config) as ctx:
                self.config = ctx.config
                self.evaluator = ctx.evaluator
                self.task = ctx.task
                self.task.setup(self.evaluator)
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

    if args.mode in ["train", "validate", "fit-scale"]:
        config = build_config(args, override_args)
        config = add_benchmark_config(config, args)
        config = add_benchmark_validate_config(config, args)
        config = add_benchmark_fit_scale_config(config, args)
    elif args.mode == "run-md":
        config = build_run_md_config(args)
    elif args.mode == "evaluate":
        config = build_evaluate_config(args)

    if args.submit:  
        # Run on cluster (using the implemented job submission)
        # Note that we did not / do not use this way for job submission.
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

    else:  
        # Run locally or cluster (using job schedulers on Samsung Supercom instead of using args.submit)
        Runner()(config)
