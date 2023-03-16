import os
import logging
import torch
import torch.distributed as dist
from ocpmodels.common import distutils as OCPdistutils

def setup(config):
    if config["submit"] or config["summit"]:
        OCPdistutils.setup(config)
    else:
        # general distributed (or multi-gpu) setting
        logging.info(f"{os.environ['LOCAL_RANK']}, {config['local_rank']}, {os.environ['RANK']}, {os.environ['WORLD_SIZE']}, {config['world_size']}, {os.environ['MASTER_ADDR']}, {os.environ['MASTER_PORT']}")
        config["local_rank"] = int(os.environ["LOCAL_RANK"])
        config["rank"] = int(os.environ["RANK"])
        config["world_size"] = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", world_size=config["world_size"], rank=config["rank"])
        torch.cuda.set_device(config["local_rank"]) # this is necessary to avoid memory leak