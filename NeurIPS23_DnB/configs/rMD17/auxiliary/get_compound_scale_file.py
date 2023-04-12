import sys
import os
import json

from glob import glob

json_dir = sys.argv[1]

mol_scale_data = []
for f in glob(json_dir + "/*"):
    data = json.load(open(f, 'r'))
    mol_scale_data.append(data)

compound_dict = {}
for k in mol_scale_data[0].keys():
    if k == "comment":
        continue
    if k == "model":
        compound_dict[k] = mol_scale_data[0][k]
    data_list = [float(data[k]) for data in mol_scale_data]
    compound_dict[k] = sum(data_list)/len(data_list)

with open(os.path.join(json_dir, "scale_file_compound.json"), "w", encoding="utf-8") as f:
    json.dump(compound_dict, f, ensure_ascii=False, indent=4)
