"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import argparse
import torch
import yaml
import os


def convert_builders(model_builders):
    builders = []
    for builder in model_builders:
        if builder == "ForceOutput":
            builders.append("StressForceOutput")
        else:
            builders.append(builder)
    return builders


def convert(ckpt_path, ckpt_name, out_path):
    # modify the config information saved in the checkpoint
    ckpt = torch.load(os.path.join(ckpt_path, ckpt_name), map_location='cpu')
    ckpt['config']['model_attributes']['regress_stress'] = True
    ckpt['config']['model_attributes']['model_builders'] = convert_builders(ckpt['config']['model_attributes']['model_builders'])
    ckpt['config']['task']['metrics'] += ['stress_mae','stress_mse']
    for k in ckpt['state_dict'].keys():
        if k.startswith("module.module"):
            add_key = ".".join(k.split(".")[:3]) + '.model._empty'
        else:
            add_key = ".".join(k.split(".")[:2]) + '.model._empty'
        break
    ckpt['state_dict'][add_key] = torch.tensor([])
    
    # modify the config file saved in the directory
    with open(os.path.join(ckpt_path, 'config_train.yml'), 'r') as f:
        config = yaml.load(f,yaml.FullLoader)
    config['model']['regress_stress'] = True
    config['model']['model_builders'] = convert_builders(config['model']['model_builders'])
    config['task']['metrics'] += ['stress_mae','stress_mse']
    config['use_stress'] = True
    os.makedirs(out_path, exist_ok=True)

    # save the results
    torch.save(ckpt, os.path.join(out_path, ckpt_name))
    with open(os.path.join(out_path, 'config_train.yml'), 'w') as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converting ForceOutput to StressForceOutput")
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Path of trained checkpoint directory",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default='checkpoint.pt',
        help="Filename of the trained checkpoint (default: checkpoint.pt)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Path of converted output checkpoint directory",
    )
    args = parser.parse_args()
    convert(args.ckpt_dir, args.ckpt_name, args.out_dir)