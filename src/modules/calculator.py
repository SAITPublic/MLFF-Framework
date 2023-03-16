
import torch
from torch_geometric.data import Data, Batch

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

from nequip.data import AtomicData, AtomicDataDict
from nequip.utils.torch_geometric.batch import Batch as BatchNequIP


def convert_atoms_to_ocp_data(atoms):
    # convert atoms into pytorch_geometric data
    # which will be converted into graph ocp data in model forward() by using generate_graph()
    atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
    positions = torch.Tensor(atoms.get_positions())
    cell = torch.Tensor(atoms.get_cell()).view(1, 3, 3)
    natoms = positions.shape[0]
    data = Data(
        cell=cell,
        pos=positions,
        atomic_numbers=atomic_numbers,
        natoms=natoms,
    ).to(self.device)
    data = Batch.from_data_list(data) # batch size = 1
    return data


def convert_atoms_to_nequip_data(atoms):
    # convert atoms into AtomicData (graph) for NequIP
    data = AtomicData.from_ase(atoms=atoms, r_max=self.cutoff)
    # remove labels
    for k in AtomicDataDict.ALL_ENERGY_KEYS:
        if k in data:
            del data[k]
    data = self.model.type_mapper(data)
    data = data.to(self.device)
    data = BatchNequIP.from_data_list(data)
    return data


class BenchmarkCalculator(Calculator):
    def __init__(self, config, model, normalizers, **kwargs):
        Calculator.__init__(self, **kwargs)

        self.config = config
        self.model = model
        self.normalizers = normalizers
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self._set_model()
        self._set_data_converter()

    def _set_model(self):
        self.model_name = self.config["model_name"]

        # evaluation mode
        self.model.eval()

        # extract arguments required to convert atom data into graph ocp data
        self.cutoff = self.model.cutoff
        self.max_neighbors = self.model.max_neighbors
        self.pbc = [self.model.use_pbc] * 3

        # because the input snapshots are new, on-the-fly graph generating 
        self.model.otf_graph = True 
    
    def _set_data_converter(self):
        if self.model_name == "nequip":
            self.convert_atoms_to_data = convert_atoms_to_nequip_data
        else:
            self.convert_atoms_to_data = convert_atoms_to_ocp_data

    def set_normalizers(self, normalizers):
        self.normalizers = normalizers

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        # atoms = ASE atoms

        # set atoms attribute
        Calculator.calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        # convert atoms (ASE) into a data format compaitible with MLFF models
        data = self.convert_atoms_to_data(atoms)

        # MLFF model inference
        energy, forces = self.model(data)

        # de-normalization (if it is used)
        if self.model_name == "nequip":
            # the model output is already in real unit in evaluation mode
            pass
        else:
            if self.normalizers is not None:
                # if normalization is used in the trainer, model output values should be de-normalized.
                # (because ground truth trained by a model was normalized)
                energy = self.normalizers["target"].denorm(energy)
                forces = self.normalizers["grad_target"].denorm(forces)
        
        # save the results
        self.results = {
            "energy": energy.item(),
            "forces": forces.cpu().numpy(),
        }