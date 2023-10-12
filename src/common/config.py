"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import os

from ocpmodels.common.utils import load_config


def add_benchmark_config(config, args):     
    # to load a checkpoint in 'timestamp_id' path and resume to train the model
    if "timestamp_id" not in config.keys():
        config["timestamp_id"] = args.timestamp_id

    # to save checkpoints at every given epoch
    config["save_ckpt_every_epoch"] = args.save_ckpt_every_epoch

    # specify resume or finetune
    config["resume"] = args.resume

    return config


def add_benchmark_fit_scale_config(config, args):
    if args.mode == "fit-scale":
        config["scale_path"] = args.scale_path
        config["scale_file"] = args.scale_file
        config["data_type"] = args.data_type
        config["num_batches"] = args.num_batches
    return config


def add_benchmark_validate_config(config, args):
    if args.mode == "validate":
        config["validate_data"] = args.validate_data
        config["validate_batch_size"] = args.validate_batch_size
        config["separate_evaluation"] = args.separate_evaluation
        config["shuffle"] = args.shuffle
    return config


def check_config(config):
    assert "task" in config
    assert "model" in config
    assert "dataset" in config
    assert "optim" in config
    assert "identifier" in config
    assert "timestamp_id" in config
    assert "is_debug" in config
    assert "print_every" in config
    assert "seed" in config
    assert "logger" in config
    assert "local_rank" in config
    assert "amp" in config
    assert "cpu" in config
    assert "noddp" in config
    assert "resume" in config

    if config.get("run_dir", "./") == "./":
        config["run_dir"] = os.getcwd()

    config["trainer"] = config.get("trainer", "forces")
    config["slurm"] = config.get("slurm", {})

    return config
    

def load_config_with_warn(config_yml, warn_string):
    if config_yml is None:
        raise Exception(warn_string)

    config, duplicates_warning, duplicates_error = load_config(config_yml)
    if len(duplicates_warning) > 0:
        logging.warning(
            f"Overwritten config parameters from included configs "
            f"(non-included parameters take precedence): {duplicates_warning}"
        )
    if len(duplicates_error) > 0:
        raise ValueError(
            f"Conflicting (duplicate) parameters in simultaneously "
            f"included configs: {duplicates_error}"
        )
    return config


def build_run_md_config(args):
    config = load_config_with_warn(
        args.md_config_yml,
        "'md-config-yml' should be given to set up a md simulation!!"
    )
    config["mode"] = args.mode
    assert args.checkpoint is not None, "--checkpoint should be given."
    config["checkpoint"] = args.checkpoint
    return config


def build_evaluate_config(args):
    if (args.evaluation_metric in ["ef", "energy_force"]
        and args.evaluation_config_yml is None):
        assert args.reference_trajectory is not None, "--reference-trajectory should be given when not using --evaluation-config-yml."
        config = {
            "reference_trajectory": args.reference_trajectory,
            "save_ef": args.save_ef,
            "measure_time": args.measure_time,
        }
    else:
        config = load_config_with_warn(
            args.evaluation_config_yml,
            "--evaluation-config-yml should be given to enable an evaluation based on simulation indicators"
        )
        
    config["mode"] = args.mode
    if args.evaluation_metric is not None:
        if "evaluation_metric" in config:
            assert config["evaluation_metric"] == args.evaluation_metric, "Please check the 'evaluation_metric' in the config file and '--evaluation-metric'"
        else:
            config["evaluation_metric"] = args.evaluation_metric
    else:
        assert "evaluation_metric" in config, "'evaluation_metric' in the config file or '--evaluation-metric' should be given."
        
    if config["evaluation_metric"] in ["ef", "energy_force", "eos", "equation_of_state", "pe_curves", "potential_energy_curves"]:
        assert args.checkpoint is not None, f"--checkpoint should be given for {config['evaluation_metric']}."
        config["checkpoint"] = args.checkpoint
    return config
