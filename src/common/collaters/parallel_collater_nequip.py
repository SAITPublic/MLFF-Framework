import torch

from nequip.utils.torch_geometric.batch import Batch as BatchNequIP
from nequip.data import AtomicData, AtomicDataDict

from src.common.utils import bm_logging # benchmark logging
from src.common.collaters.parallel_collater import ParallelCollater

# by robert.cho...
# key_mappers={
#     "pos":AtomicDataDict.POSITIONS_KEY,
#     "cell":AtomicDataDict.CELL_KEY, 
#     "atomic_numbers":AtomicDataDict.ATOMIC_NUMBERS_KEY,
#     "edge_index":AtomicDataDict.EDGE_INDEX_KEY,
#     "force":AtomicDataDict.FORCE_KEY,
#     "y":AtomicDataDict.TOTAL_ENERGY_KEY, 
#     "edge_cell_shift":AtomicDataDict.EDGE_CELL_SHIFT_KEY,
#     "fixed":"fixed",
# }
# def data_transform_oc(data: torch_geometric.data.Data ) ->  AtomicData:
#     return AtomicData(**{key_mappers[k]: data.__getattr__(k) for k in key_mappers})

def convert_ocp_Data_into_nequip_AtomicData(ocp_data, transform):
    kwargs = {}
    
    # atomic_types ???

    # atomic_numbers
    # : (nAtoms) -> (nAtoms, 1)
    kwargs["atomic_numbers"] = ocp_data.atomic_numbers.unsqueeze(-1)

    # cell
    # : (1, 3, 3) -> (3, 3)
    kwargs["cell"] = ocp_data.cell.view(3, 3)

    # edge_cell_shift
    kwargs["edge_cell_shift"] = ocp_data.cell_offsets.type(ocp_data.pos.dtype)

    # edge_index
    # : change the order of src and dest which is different from that of OCP Data
    kwargs["edge_index"] = ocp_data.edge_index[[1, 0]] 
    
    # forces
    kwargs["forces"] = ocp_data.force

    # total_energy and free_energy (which are identical for now)
    kwargs["total_energy"] = torch.tensor(ocp_data.y)
    kwargs["free_energy"] = torch.tensor(ocp_data.y)

    # pbc (deprecated)
    # : this is not used in NequIP model forward
    #kwargs["pbc"]

    # pos
    kwargs["pos"] = ocp_data.pos

    # r_max (deprecated)

    # initiate AtomicData
    data = AtomicData(**kwargs)

    # lastly, apply the transfrom defined in the type mapper
    # : atomic_numbers -> atom_types
    data = transform(data)

    # Additional information used in an OCP-based trainer
    # fixed atoms
    if hasattr(ocp_data, "fixed"):
        data.fixed = ocp_data.fixed

    return data


class ParallelCollaterNequIP(ParallelCollater):
    def __init__(self, num_gpus, otf_graph=False, transform=None, use_pbc=False, type_mapper=None):
        self.num_gpus = num_gpus
        self.otf_graph = otf_graph
        self.transform = transform # TypeMapper in NequIP
        self.use_pbc = use_pbc # torch.tensor([use_pbc] * 3)
        self.type_mapper = type_mapper

    def data_list_collater(self, data_list, otf_graph=False):
        # First, generate a list of AtomicData.
        # : from data_list (= a list of pytorch_geometric Data loaded from LMDB)
        atomic_data_list = [
            convert_ocp_Data_into_nequip_AtomicData(ocp_data=d, transform=self.type_mapper)
            for d in data_list
        ]
        # Then, the list of AtomicData will be a batch by using 'from_data_list' defined in NequIP
        # : from_data_list customized by NequIP (nequip/utils/torch_geometric/batch.py)
        batch = BatchNequIP.from_data_list(atomic_data_list)

        # convert dimension of energy in a batch to be compatible with an OCP trainer
        # (batch size, 1) -> (batch_size)
        # batch.free_energy is the shallow copy of batch.total_energy
        batch.total_energy = batch.total_energy.view(-1)

        if not otf_graph:
            batch = self.set_neighbors_in_a_batch(data_list, batch)
        return batch