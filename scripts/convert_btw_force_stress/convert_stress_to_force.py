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
        if builder == "StressForceOutput":
            builders.append("ForceOutput")
        else:
            builders.append(builder)
    return builders


def convert(ckpt_path, ckpt_name, out_path):
    # modify the config information saved in the checkpoint
    ckpt = torch.load(os.path.join(ckpt_path, ckpt_name), map_location='cpu')
    ckpt['config']['model_attributes']['regress_stress'] = False
    ckpt['config']['model_attributes']['model_builders'] = convert_builders(ckpt['config']['model_attributes']['model_builders'])
    ckpt['config']['task']['metrics'] = ['energy_per_atom_mae', 'energy_per_atom_mse', 'forces_mae', 'forces_mse']
    for k in ckpt['state_dict'].keys():
        if k.startswith("module.module"):
            del_key = ".".join(k.split(".")[:3]) + '.model._empty'
        else:
            del_key = ".".join(k.split(".")[:2]) + '.model._empty'
        break
    ckpt['state_dict'].pop(del_key)
    
    # modify the config file saved in the directory
    with open(os.path.join(ckpt_path, 'config_train.yml'),'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config['model']['regress_stress'] = False
    config['model']['model_builders'] = convert_builders(config['model']['model_builders'])
    config['task']['metrics'] = ['energy_per_atom_mae', 'energy_per_atom_mse', 'forces_mae', 'forces_mse']
    config['use_stress'] = False
    os.makedirs(out_path, exist_ok=True)

    # save the results
    torch.save(ckpt, os.path.join(out_path, ckpt_name))
    with open(os.path.join(out_path, 'config_train.yml'), 'w') as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converting StressForceOutput to ForceOutput")
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