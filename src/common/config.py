import os

from ocpmodels.common.utils import load_config


def add_benchmark_config(config, args):     
    # to load a checkpoint in 'timestamp_id' path and resume to train the model
    if "timestamp_id" not in config.keys():
        config["timestamp_id"] = args.timestamp_id

    # to save checkpoints at every given epoch
    config["save_ckpt_every_epoch"] = args.save_ckpt_every_epoch

    # to load a molecule dataset of rMD17
    if args.molecule is not None:
        if args.molecule in config["dataset"].keys():
            config["dataset"] = config["dataset"][args.molecule]
        else:
            raise ValueError(f"The path of {args.molecule} dataset is not specified in your configuration file")

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

    if config.get("run_dir", "./") == "./":
        config["run_dir"] = os.getcwd()

    config["trainer"] = config.get("trainer", "forces")
    config["slurm"] = config.get("slurm", {})

    return config
    

def load_md_config(md_config_yml):
    if md_config_yml is None:
        return {}

    config, duplicates_warning, duplicates_error = load_config(md_config_yml)
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
    config = load_md_config(args.md_config_yml)
    config["mode"] = args.mode
    assert args.checkpoint is not None, "--checkpoint is should be given."
    config["checkpoint"] = args.checkpoint
    config["initial_structure"] = args.initial_structure
    config["output_trajectory"] = args.output_trajectory
    return config


def build_evaluate_config(args):
    config = load_md_config(args.md_config_yml)
    config["mode"] = args.mode
    assert args.evaluation_metric is not None, "--evaluation-metric is should be given."
    config["evaluation_metric"] = args.evaluation_metric
    assert args.checkpoint is not None, "--checkpoint is should be given."
    config["checkpoint"] = args.checkpoint
    config["reference_trajectory"] = args.reference_trajectory
    config["generated_trajectory"] = args.generated_trajectory
    return config
