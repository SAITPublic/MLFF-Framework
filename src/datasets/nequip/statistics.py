import os
import numpy as np
import torch

from nequip.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from nequip.utils.torch_geometric.batch import Batch as BatchNequIP
from nequip.data.transforms import TypeMapper

from ocpmodels.common import distutils

from src.common.utils import bm_logging # benchmark logging
from src.common.collaters.parallel_collater_nequip import convert_ocp_Data_into_nequip_AtomicData


RESCALE_THRESHOLD = 1e-6


def helper_avg_num_neighbors(data):
    counts = torch.unique(
        data[AtomicDataDict.EDGE_INDEX_KEY][0],
        sorted=True,
        return_counts=True,
    )[1]
    # in case the cutoff is small and some nodes have no neighbors,
    # we need to pad `counts` up to the right length
    counts = torch.nn.functional.pad(
        counts, pad=(0, len(data[AtomicDataDict.POSITIONS_KEY]) - len(counts))
    )
    return (counts, "node")


# reference: nequip.model.builder_utils.py
def compute_avg_num_neighbors(config, initialize, dataset, transform):
    # Compute avg_num_neighbors
    ann = config.get("avg_num_neighbors", "auto")
    if ann == "auto":
        if not initialize:
            # raise RuntimeError("avg_num_neighbors = auto but initialize is False")
            # TODO: exclude the avg_num_neighbors computation when loading a checkpoint
            bm_logging.info("When `initialize` is set as False, the avg_num_neighbors will be loaded from the checkpoint")
            bm_logging.ifno("But, for now, you need to a file which includes avg_num_neighbors")
        if dataset is None:
            raise RuntimeError("When avg_num_neighbors = auto, the dataset is required")
        _mean, _std = statistics(
            dataset=dataset,
            transform=transform,
            fields=[helper_avg_num_neighbors],
            modes=["mean_std"],
            stride=config.get("dataset_statistics_stride", 1),
        )[0]
        _mean = float(_mean.item())
        return _mean
    else:
        assert isinstance(ann, float), "avg_num_neighbors should be float, if not set as auto"
        return ann


# reference: nequip.model.rescale.py (GlobalRescale)
def compute_global_shift_and_scale(config, initialize, dataset, transform):
    global_shift = config.get("global_rescale_shift", None)
    global_scale = config.get(
        "global_rescale_scale", 
        f"dataset_{AtomicDataDict.FORCE_KEY}_rms" # default string when training forces
    )

    if global_shift is not None:
        default_shift_keys = [AtomicDataDict.TOTAL_ENERGY_KEY]
        bm_logging.warning(
            f"!!!! Careful global_shift is set to {global_rescale_shift}."
            f"The model for {default_shift_keys} will no longer be system-size extensive"
        )

    if initialize:
        str_names = []
        for value in [global_shift, global_scale]:
            if isinstance(value, str):
                str_names.append(value)
            elif (
                value is None
                or isinstance(value, float)
                or isinstance(value, torch.Tensor)
            ):
                pass
            else:
                raise ValueError(f"Invalid global scale `{value}`")
            
        if len(str_names) > 0:
            stats = compute_stats(
                str_names=str_names,
                dataset=dataset,
                transform=transform,
                stride=config.get("dataset_statistics_stride", 1),
            )
        
        if isinstance(global_shift, str):
            s = global_shift
            global_shift = stats[str_names.index(global_shift)]

        if isinstance(global_scale, str):
            s = global_scale
            global_scale = stats[str_names.index(global_scale)]
            if global_scale < RESCALE_THRESHOLD:
                raise ValueError(
                    f"Global energy scaling was very low: {global_scale}. If dataset values were used, does the dataset contain insufficient variation? Maybe try disabling global scaling with global_scale=None."
                )
        
        bm_logging.info(
            f"[global rescale] Globally, energy and forces are scaled by: {global_scale}, and energy is shifted by {global_shift}."
        )
    else:
        # put dummy node
        global_shift = 0.0 if global_shift is not None else None
        global_scale = 1.0 if global_scale is not None else None
        bm_logging.info("[global rescale] When `initialize` is set as False, the global scale and shift will be loaded from the checkpoint")
    
    return global_shift, global_scale


# reference: nequip.model.rescale.py (PerSpeciesRescale)
def compute_per_species_shift_and_scale(config, initialize, dataset, transform):
    scales = config.get(
        "per_species_rescale_scales",
        f"dataset_{AtomicDataDict.FORCE_KEY}_rms"
        if AtomicDataDict.FORCE_KEY in config.get("train_on_keys", [])
        else f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_std"
    )
    shifts = config.get(
        "per_species_rescale_shifts",
        f"dataset_per_atom_{AtomicDataDict.TOTAL_ENERGY_KEY}_mean"
    )

    # Check for common double shift mistake with defaults
    if "RescaleEnergyEtc" in config.get("model_builders", []):
        if (config.get("global_rescale_shift", None) is not None
            and shifts is not None
        ):
            raise RuntimeError(
                "A global_rescale_shift was provided, but the default per-atom energy shift was not disabled (one of them should be disabled)."
            )
    
    arguments_in_dataset_units = False
    if initialize:
        str_names = []
        for value in [shifts, scales]:
            if isinstance(value, str):
                str_names += [value]
            elif (
                value is None
                or isinstance(value, float)
                or isinstance(value, list)
                or isinstance(value, torch.Tensor)
            ):
                # valid values
                pass
            else:
                raise ValueError(f"Invalid value `{value}` of type {type(value)}")

        if len(str_names) == 2:
            # Both computed from dataset
            arguments_in_dataset_units = True
        elif len(str_names) == 1:
            if None in [shifts, scales]:
                arguments_in_dataset_units = True
            else:
                assert config[
                    "per_species_rescale_arguments_in_dataset_units"
                ], "Requested to set either the shifts or scales of the per_species_rescale using dataset values, but chose to provide the other in non-dataset units. Please give the explictly specified shifts/scales in dataset units and set per_species_rescale_arguments_in_dataset_units"

        if len(str_names) > 0:
            stats = compute_stats(
                str_names=str_names,
                dataset=dataset,
                transform=transform,
                stride=config.dataset_statistics_stride,
                kwargs=config.get("per_species_rescale_kwargs", {}),
            )

        if isinstance(shifts, str):
            s = shifts
            shifts = stats[str_names.index(shifts)].squeeze(-1)  # energy is 1D
        elif isinstance(shifts, (list, float)):
            shifts = torch.as_tensor(shifts)

        if isinstance(scales, str):
            s = scales
            scales = stats[str_names.index(scales)].squeeze(-1)  # energy is 1D
        elif isinstance(scales, (list, float)):
            scales = torch.as_tensor(scales)

        if scales is not None and torch.min(scales) < RESCALE_THRESHOLD:
            raise ValueError(
                f"Per species energy scaling was very low: {scales}. Maybe try setting {module_prefix}_scales = 1."
            )
        
        bm_logging.info(
            f"[per species rescale] Atomic outputs are scaled by: {TypeMapper.format(scales, config.type_names)}, shifted by {TypeMapper.format(shifts, config.type_names)}."
        )
    else:
        # Put dummy values
        scales = 1.0 if scales is not None else None
        shifts = 0.0 if shifts is not None else None
        bm_logging.info("[per species rescale] When `initialize` is set as False, the per species scales and shifts will be loaded from the checkpoint")

    return shifts, scales, arguments_in_dataset_units


# reference: nequip.data.dataset.py implemented by robert.cho
#            (https://github.sec.samsung.net/ESC-MLFF/NequIP/blob/OC22/nequip/data/dataset.py)
# Note: In NequIP, this dataset consists of batchs 
#                 (which are generated by Batch.from_data_list at the preprocessing phase).
def _per_atom_statistics(
    ana_mode: str,
    arr: torch.Tensor,
    batch: torch.Tensor,
    unbiased: bool = True,
):
    """Compute "per-atom" statistics that are normalized by the number of atoms in the system.
    Only makes sense for a graph-level quantity (checked by .statistics).
    """
    # using unique_consecutive handles the non-contiguous selected batch index
    _, N = torch.unique_consecutive(batch, return_counts=True)
    N = N.unsqueeze(-1)
    assert N.ndim == 2
    assert N.shape == (len(arr), 1)
    assert arr.ndim >= 2
    data_dim = arr.shape[1:]
    arr = arr / N
    assert arr.shape == (len(N),) + data_dim
    if ana_mode == "mean_std":
        mean = torch.mean(arr, dim=0)
        std = torch.std(arr, unbiased=unbiased, dim=0)
        mean_square = torch.mean(arr*arr, dim=0)
        return mean, std, mean_square
    elif ana_mode == "rms":
        return (torch.sqrt(torch.mean(arr.square())),)
    else:
        raise NotImplementedError(
            f"{ana_mode} for per-atom analysis is not implemented"
        )


def statistics(dataset, transform, fields, modes, stride, unbiased=True, kwargs={}):
    assert len(modes) == len(fields)
    assert transform is not None

    if len(fields) == 0:
        return []

    if isinstance(fields[0], str):
        filepath = "NequIP_statistics-" + ",".join(modes) + "-" + ",".join(fields) + ".pt"
    else:
        filepath = "NequIP_statistics-" + ",".join(modes) + "-num_neighbors.pt"

    # dataset.path is pathlib.Path
    if dataset.path.is_dir():
        filepath = dataset.path / filepath
    elif dataset.path.is_file():
        filepath = dataset.path.parent / filepath
    else:
        RuntimeError(f"Check dataset path (given : {type(dataset.path)} - {dataset.path})")

    if filepath.exists():
        out = torch.load(filepath)
        bm_logging.info(f"Statistics loaded from {filepath} (stats: {out})")
        return out
    else:
        bm_logging.info(f"Start computing statistics which will be saved at {filepath}")

    # dataset (LMDB) -> data_list -> convert it into the batch using Batch.from_data_list
    # -> convert the batch into AtomicData batch using NequIP version of from_data_list
    bs = 32
    sz = (len(dataset)-1)//bs + 2
    idxs=[bs*i if bs*i<=len(dataset) else len(dataset) for i in np.arange(sz)]

    vals = {}
    n = {}
    for i in range(sz - 1):
        data_list = [dataset[j] for j in np.arange(idxs[i], idxs[i+1])]
        atomic_data_list = [
            convert_ocp_Data_into_nequip_AtomicData(d, transform=transform)
            for d in data_list
        ]
        data = BatchNequIP.from_data_list(atomic_data_list) # data = Batch of AtomicData
        
        num_graphs = data.batch.max()+1
        graph_selector = torch.arange(0, num_graphs, stride)
        node_selector = torch.as_tensor(
            np.in1d(data.batch.numpy(), graph_selector.numpy())
        )
        num_nodes = node_selector.sum()
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_selector = node_selector[edge_index[0]] & node_selector[edge_index[1]]
        num_edges = edge_selector.sum()
        del edge_index

        #ff_transformed = [] 

        if transform is None:
            data_transformed = data.to_dict()
        else:
            data_transformed = transform(data.to_dict(), types_required=False)

        selectors = {}
        for k in data_transformed.keys():
            if k in _NODE_FIELDS:
                selectors[k] = node_selector
            elif k in _GRAPH_FIELDS:
                selectors[k] = graph_selector
            elif k == AtomicDataDict.EDGE_INDEX_KEY:
                selectors[k] = (slice(None, None, None), edge_selector)
            elif k in _EDGE_FIELDS:
                selectors[k] = edge_selector
        data_transformed = {
            k: data_transformed[k][selectors[k]]
            for k in data_transformed.keys()
            if k in selectors
        }

        for ifield, field in enumerate(fields):
            if callable(field):
                arr, arr_is_per = field(data_transformed)
                arr = arr.to(torch.get_default_dtype())  # all statistics must be on floating
                assert arr_is_per in ("node", "graph", "edge")
            else:
                if field not in data_transformed:
                    raise RuntimeError(
                    f"Field `{field}` for which statistics were requested not found in data."
                )
                if field not in selectors:
                    raise RuntimeError(
                    f"Only per-node and per-graph fields can have statistics computed; `{field}` has not been registered as either. If it is per-node or per-graph, please register it as such using `nequip.data.register_fields`"
                )
                arr = data_transformed[field]
                if field in _NODE_FIELDS:
                    arr_is_per = "node"
                elif field in _GRAPH_FIELDS:
                    arr_is_per = "graph"
                elif field in _EDGE_FIELDS:
                    arr_is_per = "edge"
                else:
                    raise RuntimeError

            # Check arr
            if arr is None:
                raise ValueError(
                    f"Cannot compute statistics over field `{field}` whose value is None!"
                )
            if not isinstance(arr, torch.Tensor):
                if np.issubdtype(arr.dtype, np.floating):
                    arr = torch.as_tensor(arr, dtype=torch.get_default_dtype())
                else:
                    arr = torch.as_tensor(arr)
            if arr_is_per == "node":
                arr = arr.view(num_nodes, -1)
            elif arr_is_per == "graph":
                arr = arr.view(num_graphs, -1)
            elif arr_is_per == "edge":
                arr = arr.view(num_edges, -1)

            ana_mode = modes[ifield]
            if ana_mode == "count":
                raise NotImplementedError(f"Not implemented")
            elif ana_mode == "rms":
                sqr_sum = torch.sum(arr * arr)
                n_now = np.prod(arr.shape)
                if field not in vals.keys():
                    vals[field] = torch.sqrt(torch.mean(arr * arr)).reshape(1)
                    n[field] = 0
                else:
                    rms_origin = vals[field]
                    n_origin = n[field]
                    n_tot = n_now + n_origin                        
                    vals[field] = torch.sqrt((rms_origin*rms_origin*(n_origin/n_tot) + sqr_sum/n_tot))
                n[field] += n_now
            elif ana_mode == "mean_std":
                mean = torch.mean(arr, dim=0)
                std = torch.std(arr, dim=0, unbiased=unbiased)
                mean_square = torch.mean(arr*arr, dim=0)
                if field not in vals.keys():
                    vals[field] = (torch.cat((mean,std)))
                    n[field] = 0
                else:
                    std_origin = vals[field][1]
                    mean_origin = vals[field][0] + 0
                    n_origin = n[field]
                    n_now = arr.shape[0]
                    n_tot = n_now + n_origin                        
                    vals[field][0] = (mean_origin*n[field]/n_tot+mean*n_now/n_tot)
                    vals[field][1] = (std_origin*std_origin*((n_origin-1)/(n_tot-1)) +mean_origin*mean_origin*(n_origin/(n_tot-1))+mean_square*(n_now/(n_tot-1)))
                    vals[field][1] = torch.sqrt(vals[field][1]-vals[field][0]*vals[field][0]*(n_tot/(n_tot-1)))
                n[field] += arr.shape[0]
            elif ana_mode.startswith("per_species_"):
                raise NotImplementedError(f"Not implemented for {ana_mode}")
            elif ana_mode.startswith("per_atom_"):
                # per-atom
                # only makes sense for a per-graph quantity
                if arr_is_per != "graph":
                    raise ValueError(
                    f"It doesn't make sense to ask for `{ana_mode}` since `{field}` is not per-graph"
                )
                ana_mode = ana_mode.replace("per_atom_", "")
                results = _per_atom_statistics(
                    ana_mode=ana_mode,
                    arr=arr,
                    batch=data_transformed[AtomicDataDict.BATCH_KEY],
                    unbiased=unbiased,
                )
                if ana_mode == "mean_std":
                    mean, std, mean_square = results
                    if field not in vals.keys():
                        vals[field] = (torch.cat((mean,std)))
                        n[field] = 0
                    else:
                        std_origin = vals[field][1]
                        mean_origin = vals[field][0]+0
                        n_origin = n[field]
                        n_now = arr.shape[0]
                        n_tot = n_now+n_origin                        
                        vals[field][0] = mean_origin*(n[field]/n_tot)+mean*(n_now/n_tot)
                        vals[field][1] = std_origin*std_origin*((n_origin-1)/(n_tot-1)) +mean_origin*mean_origin*(n_origin/(n_tot-1))+mean_square*(n_now/(n_tot-1))
                        vals[field][1] = torch.sqrt(vals[field][1]-vals[field][0]*vals[field][0]* (n_tot/(n_tot-1)))
                    n[field] += arr.shape[0]
                else:
                    raise NotImplementedError(f"Not implemented")
            else:
                raise NotImplementedError(f"Cannot handle statistics mode {ana_mode}")
        
    out = []
    for field in fields:
        out.append(tuple(val for val in vals[field]))

    torch.save(out, filepath)
    bm_logging.info(f"Computing statistics is done (stats: {out}), and saved at {filepath}")
    return out


def compute_stats(str_names, dataset, transform, stride, kwargs={}):
    # parse the list of string to field, mode
    # and record which quantity correspond to which computed_item
    stat_modes = []
    stat_fields = []
    stat_strs = []
    ids = []
    tuple_ids = []
    tuple_id_map = {"mean": 0, "std": 1, "rms": 0}
    input_kwargs = {}
    for name in str_names:
        # remove dataset prefix
        name = name.replace("dataset_", "")

        # identify per_species and per_atom modes
        prefix = ""
        if name.startswith("per_species_"):
            name = name.replace("per_species_", "")
            prefix = "per_species_"
        elif name.startswith("per_atom_"):
            name = name.replace("per_atom_", "")
            prefix = "per_atom_"

        stat  = name.split("_")[-1]
        field = "_".join(name.split("_")[:-1])
        if stat in ["mean", "std"]:
            stat_mode = prefix + "mean_std"
            stat_str = field + prefix + "mean_std"
        elif stat in ["rms"]:
            stat_mode = prefix + "rms"
            stat_str = field + prefix + "rms"
        else:
            raise ValueError(f"Cannot handle {stat} type quantity")

        if stat_str in stat_strs:
            ids += [stat_strs.index(stat_str)]
        else:
            ids += [len(stat_strs)]
            stat_strs += [stat_str]
            stat_modes += [stat_mode]
            stat_fields += [field]
            if stat_mode.startswith("per_species_"):
                if field in kwargs:
                    input_kwargs[field + stat_mode] = kwargs[field]
        tuple_ids += [tuple_id_map[stat]]

    values = statistics(
        dataset=dataset,
        transform=transform,
        fields=stat_fields,
        modes=stat_modes,
        stride=stride,
        kwargs=input_kwargs,
    )
    return [values[idx][tuple_ids[i]] for i, idx in enumerate(ids)]