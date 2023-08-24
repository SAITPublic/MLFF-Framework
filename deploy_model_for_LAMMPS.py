import sys
import os
sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(1, os.path.abspath("./codebases/ocp")) 
sys.path.insert(2, os.path.abspath("./codebases/nequip")) 
sys.path.insert(3, os.path.abspath("./codebases/allegro")) 

import torch
import ase
import numpy as np
from ocpmodels.common.utils import load_state_dict
from src.models.nequip.nequip import NequIPWrap
from src.models.allegro.allegro import AllegroWrap

from e3nn.util.jit import script

from typing import Final

CONFIG_KEY: Final[str] = "config"
NEQUIP_VERSION_KEY: Final[str] = "nequip_version"
TORCH_VERSION_KEY: Final[str] = "torch_version"
E3NN_VERSION_KEY: Final[str] = "e3nn_version"
CODE_COMMITS_KEY: Final[str] = "code_commits"
R_MAX_KEY: Final[str] = "r_max"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_BAILOUT_KEY: Final[str] = "_jit_bailout_depth"
JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
TF32_KEY: Final[str] = "allow_tf32"

"""
Usage:
python deploy_model_for_LAMMPS.py NequIP [a checkpoint of NequIP] 
python deploy_model_for_LAMMPS.py Allegro [a checkpoint of Allegro] 
"""

def compile_for_deploy(model):
    model.eval()
    if not isinstance(model, torch.jit.ScriptModule):
        model = script(model)
    return model

ckpt_path = sys.argv[2]
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt["config"]["model_attributes"]["initialize"] = False

new_dict ={}
for k, v in ckpt["state_dict"].items():
    while k.startswith("module."):
        k = k[7:]
    new_dict[k] = v

if sys.argv[1] == "NequIP":
    model = NequIPWrap(num_atoms=None, bond_feat_dim=None, num_targets=1, **ckpt["config"]["model_attributes"])
    load_state_dict(model, new_dict, strict=True)
    model = compile_for_deploy(model.nequip_model)
elif sys.argv[1] == "Allegro":
    model = AllegroWrap(num_atoms=None, bond_feat_dim=None, num_targets=1, **ckpt["config"]["model_attributes"])
    load_state_dict(model, new_dict, strict=True)
    model = compile_for_deploy(model.allegro_model)
else:
    raise RunError("other models are not supported")

# Deploy
metadata = {}
metadata[TORCH_VERSION_KEY] = "1.12.1+cu116"
metadata[E3NN_VERSION_KEY] = "0.5.1"
metadata[NEQUIP_VERSION_KEY] = "0.5.6"
metadata[R_MAX_KEY] = "6.0"
if('chemical_symbols' in ckpt['config']['model_attributes'].keys()):
    chemical_symbols=ckpt['config']['model_attributes']['chemical_symbols']
    atomic_nums = [ase.data.atomic_numbers[sym] for sym in chemical_symbols]
    chemical_symbols = [
                    e[1] for e in sorted(zip(atomic_nums, chemical_symbols))
                ]
    type_names = chemical_symbols
elif ('chemical_symbol_to_type' in ckpt['config']['model_attributes'].keys()):
    symbols=list(ckpt['config']['model_attributes']['chemical_symbol_to_type'].keys())
    vals=[ ckpt['config']['model_attributes']['chemical_symbol_to_type'][sym_] for sym_ in symbols]
    idx=np.argsort(vals)
    sym__=np.array(symbols)
    type_names=list(sym__[idx])
metadata[N_SPECIES_KEY] = f"{len(type_names)}"
metadata[TYPE_NAMES_KEY] = " ".join(type_names)
metadata[JIT_BAILOUT_KEY] = "2"
metadata[JIT_FUSION_STRATEGY] = "DYNAMIC,3"
metadata[TF32_KEY] = "0" # false
metadata[CONFIG_KEY] = "" # yaml.dump(dict(config))

metadata = {k: v.encode("ascii") for k, v in metadata.items()}
torch.jit.save(model, str(ckpt_path).replace(".pt", "_deployed.pt"), _extra_files=metadata)

