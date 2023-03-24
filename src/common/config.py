import os


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


def add_benchmark_evaluate_config(config, args):
    if args.mode == "evaluate":
        #config["checkpoint"] = args.checkpoint
        assert config["checkpoint"] is not None
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
    