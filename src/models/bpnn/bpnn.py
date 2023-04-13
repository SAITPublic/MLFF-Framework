"""
written by robert.cho and byunggook.na (SAIT)
"""

import os
import numpy as np
from pathlib import Path

# from ase.atom import atomic_numbers
import ase


import torch
from torch.nn import Sequential, ModuleDict
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.data import Batch
from scipy.linalg import eigh
from ocpmodels.datasets import LmdbDataset ## TODO: compute in advance?
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.gemnet.utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)
from ocpmodels.common import distutils

from src.common.utils import bm_logging # benchmark logging


class PerAtomFCN(torch.nn.Module):
    def __init__(
        self,
        nDescriptor: int,
        nHidden: int,
        nLayer: int,
        atom_types,
    ):
        super().__init__()

        self.atom_types = atom_types
        module_dict = {}
        for key in atom_types:
            module_list = []
            for j in range(nLayer):
                if j == 0:
                    module_list.append(torch.nn.Linear(nDescriptor,nHidden))
                else:
                    module_list.append(torch.nn.Linear(nHidden,nHidden))
                torch.manual_seed(0)
                torch.nn.init.xavier_normal_(module_list[-1].weight)
                tmp_bias = torch.zeros([1, module_list[-1].bias.size(0)])
                torch.nn.init.xavier_normal_(tmp_bias)
                module_list[-1].bias.data = tmp_bias[0]
                module_list.append(torch.nn.Sigmoid())
            module_list.append(torch.nn.Linear(nHidden,1))
            torch.manual_seed(0)
            torch.nn.init.xavier_normal_(module_list[-1].weight)
            tmp_bias = torch.zeros([1, module_list[-1].bias.size(0)])
            torch.nn.init.xavier_normal_(tmp_bias)
            module_list[-1].bias.data = tmp_bias[0]

            module_dict[str(key)] = Sequential(*module_list)
        self.fcn_atoms = ModuleDict(module_dict)

    def forward(self, x):
        out={key: self.fcn_atoms[str(key)](x[key]) for key in x.keys()}
        return out


class ACSF(torch.nn.Module):
    def __init__(
        self,
        atomic_numbers,
        cutoff: float = 6.0,
        use_pbc: bool = True,
        trainable=False,
        g2_params=None,
        g4_params=None,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.atomic_numbers = atomic_numbers

        if g2_params is None:
            # default G2 parameters
            g2_etas = torch.tensor([0.003214,0.035711,0.071421,0.124987,0.214264,0.357106,0.714213,1.428426])
        else:
            with open(g2_params,'r') as f:
                lines = f.readlines()
                g2_etas = torch.tensor([eval(v) for v in lines[0].split(",")])

        if g4_params is None:
            # default G4 parameters
            etas = torch.tensor([0.000357,0.028569,0.089277])
            zetas = torch.tensor([1.0,2.0,4.0])
            lmdas = torch.tensor([-1.0,1.0])
        else:
            with open(g4_params,'r') as f:
                lines = f.readlines()
                etas = torch.tensor([eval(v) for v in lines[0].split(",")])
                zetas = torch.tensor([eval(v) for v in lines[1].split(",")])
                lmdas = torch.tensor([eval(v) for v in lines[2].split(",")])

        n_species = len(self.atomic_numbers)
        self.dim_descriptor = n_species*len(g2_etas) + n_species*(n_species+1)*len(etas)*len(zetas)*len(lmdas)//2
        self.G2_params = {}
        self.G4_params_etas = {}
        self.G4_params_zetas = {}
        self.G4_params_lmdas = {}
        for atom1 in self.atomic_numbers:
            for i, atom2 in enumerate(self.atomic_numbers):
                self.G2_params[str((atom1,atom2))] = torch.nn.Parameter(g2_etas.clone().detach(), requires_grad=trainable)
                for j in range(i, len(self.atomic_numbers)):
                    atom3 = self.atomic_numbers[j]
                    self.G4_params_etas[str((atom1,atom2,atom3))] = torch.nn.Parameter(etas.clone().detach(), requires_grad=trainable)
                    self.G4_params_zetas[str((atom1,atom2,atom3))] = torch.nn.Parameter(zetas.clone().detach(), requires_grad=trainable)
                    self.G4_params_lmdas[str((atom1,atom2,atom3))] = torch.nn.Parameter(lmdas.clone().detach(), requires_grad=trainable)
        self.G2_params       = torch.nn.ParameterDict(self.G2_params)
        self.G4_params_etas  = torch.nn.ParameterDict(self.G4_params_etas)
        self.G4_params_zetas = torch.nn.ParameterDict(self.G4_params_zetas)
        self.G4_params_lmdas = torch.nn.ParameterDict(self.G4_params_lmdas)

    def forward(self, atomic_numbers, edge_index, D_st, id3_ba, id3_ca, cosφ_cab):
        eba_idx = id3_ba[id3_ba>id3_ca]
        eca_idx = id3_ca[id3_ba>id3_ca]
        cos_idx = cosφ_cab[id3_ba>id3_ca]
        D_ba = D_st[eba_idx]
        D_ca = D_st[eca_idx]

        atom_idx = {atom: atomic_numbers == atom for atom in self.atomic_numbers}
        atom_exist = {atom: atom_idx[atom].sum() > 0 for atom in self.atomic_numbers}

        res_G4 = {atom: [] for atom in self.atomic_numbers}
        for atoms_triplet in self.G4_params_etas.keys():
            atom_a, atom_b, atom_c = eval(atoms_triplet)
            eta = self.G4_params_etas[atoms_triplet]
            zeta = self.G4_params_zetas[atoms_triplet]
            lmda = self.G4_params_lmdas[atoms_triplet]
            idx = ((atomic_numbers[edge_index[:,eba_idx][0,:]] == atom_b)
                    *(atomic_numbers[edge_index[:,eba_idx][1,:]] == atom_a)
                    *(atomic_numbers[edge_index[:,eca_idx][0,:]] == atom_c)) + \
                  ((atomic_numbers[edge_index[:,eba_idx][0,:]] == atom_c)
                    *(atomic_numbers[edge_index[:,eba_idx][1,:]] == atom_a)
                    *(atomic_numbers[edge_index[:,eca_idx][0,:]] == atom_b)) 
            R_ba = D_ba[idx]
            R_ca = D_ca[idx]
            R_bc = (R_ba*R_ba + R_ca*R_ca-2*R_ba*R_ca*cos_idx[idx]).sqrt()
            cutoff = (
                (0.5*torch.cos(torch.pi*R_ca/self.cutoff) + 0.5) * 
                (0.5*torch.cos(torch.pi*R_bc/self.cutoff) + 0.5) * 
                (0.5*torch.cos(torch.pi*R_ba/self.cutoff) + 0.5) * 
                (R_bc<self.cutoff) * 
                (R_ba<self.cutoff) * 
                (R_ca<self.cutoff)
            )
            rad = torch.exp(-eta.reshape(1,-1)*(R_bc*R_bc + R_ba*R_ba + R_ca*R_ca).reshape(-1,1))
            cos_lmda = torch.pow(torch.abs(1+lmda.reshape(1,-1,1)*cos_idx[idx].reshape(-1,1,1)), zeta.reshape(1,1,-1))
            res = cos_lmda.reshape(cos_lmda.shape[0],1,cos_lmda.shape[1],cos_lmda.shape[2]) * rad.reshape(rad.shape+(1,1)) * cutoff.reshape(-1,1,1,1)
            g4_per_atom = torch.zeros((atomic_numbers.shape[0],)+res.shape[1:], device=res.device)
            scatter(res, edge_index[:,eba_idx[idx]][1,:], dim=0, out=g4_per_atom)
            if atom_exist[atom_a]:
                res_G4[atom_a].append(torch.pow(2, 1-zeta) * g4_per_atom[atom_idx[atom_a]])

        res_G2 = {atom: [] for atom in self.atomic_numbers}        
        for atoms_pair, eta in self.G2_params.items():
            source_atom, target_atom = eval(atoms_pair)
            idx = (atomic_numbers[edge_index[1,:]] == target_atom) * (atomic_numbers[edge_index[0,:]] == source_atom)
            cutoff = 0.5*torch.cos(torch.pi*D_st[idx]/self.cutoff) + 0.5
            # eta = eta.to(D_st.device)
            rad = torch.exp(-(eta).reshape(1,-1) * (D_st[idx]*D_st[idx]).reshape(-1,1))
            res = cutoff.reshape(-1,1) * rad
            out_per_atom = torch.zeros(atomic_numbers.shape[0], len(eta), device=res.device)
            scatter(res, edge_index[1,idx], dim=0, out=out_per_atom)
            if atom_exist[target_atom]:
                res_G2[target_atom].append(out_per_atom[atom_idx[target_atom]])

        non_empty_atoms = []
        for atom, val in res_G4.items():
            if len(val) > 0:
                val = torch.stack(val,dim=-1)
                val = val.reshape(val.shape[0],-1)
                res_G4[atom] = val
                non_empty_atoms.append(atom)
        for atom, val in res_G2.items():
            if len(val) > 0:
                val = torch.stack(val,dim=-1)
                val = val.reshape(val.shape[0],-1)
                res_G2[atom] = val
        res = {atom: torch.cat([res_G2[atom],res_G4[atom]], dim=-1) for atom in non_empty_atoms}
        return res


# some functions are defined in GemNet (ocpmodels.models.gemnet.gemnet.py)
@registry.register_model("bpnn")
class BPNN(BaseModel):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        cutoff=6.0,
        max_neighbors=50,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        direct_forces=False,
        # BPNN arguments
        atom_species=None,
        scale_pca_path=None,
        dataset_path=None,
        trainable=False,
        nHidden=1,
        nLayer=1,
        g2_params=None,
        g4_params=None,
    ):
        assert atom_species is not None

        super().__init__()

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc

        # convert atom_species into atomic numbers using ase.atom.atomic_numbers
        self.atomic_numbers = sorted([ase.atom.atomic_numbers[atom] for atom in atom_species])

        self.descriptor = ACSF(
            atomic_numbers=self.atomic_numbers,
            cutoff=self.cutoff,
            use_pbc=use_pbc,
            trainable=trainable,
            g2_params=g2_params,
            g4_params=g4_params,
        )

        self.per_atom_fcn = PerAtomFCN(
            nDescriptor=self.descriptor.dim_descriptor,
            nHidden=nHidden,
            nLayer=nLayer,
            atom_types=self.atomic_numbers,
        )

        if scale_pca_path is None and dataset_path is None:
            bm_logging.warning("scale and pca are not used (if they are used, BPNN accuarcy could be improved)")
            self.scale = None
            self.pca = None
        else:
            # set scale and pca
            if scale_pca_path is None:
                scale_pca_path = Path(dataset_path).parent / "BPNN_scale_pca.pt"

            if os.path.exists(scale_pca_path):
                scale_pca = torch.load(scale_pca_path)
                scale = scale_pca["scale"]
                pca = scale_pca["pca"]
                bm_logging.info(f"scale and pca are loaded from {scale_pca_path}")
            else:
                assert dataset_path is not None
                scale, pca = self._calculate_scale_and_fit_pca(
                    dataset=LmdbDataset({'src': dataset_path}), 
                    bs=256,
                )
                torch.save({"scale": scale, "pca": pca}, scale_pca_path)
                bm_logging.info(f"scale and pca are calculated and saved at {scale_pca_path}")
            self.scale = scale
            self.pca = pca

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.
        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(
            idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype
        )
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.
        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def select_edges(
        self,
        data,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        cutoff=None,
    ):
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_interaction_graph(self, data):
        num_atoms = data.atomic_numbers.size(0)
        (
            edge_index,
            D_st,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)
        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        V_st = -distance_vec / D_st[:, None]

        # Mask interaction edges if required
        if self.otf_graph or np.isclose(self.cutoff, 6):
            select_cutoff = None
        else:
            select_cutoff = self.cutoff
        (edge_index, cell_offsets, neighbors, D_st, V_st,) = self.select_edges(
            data=data,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            edge_dist=D_st,
            edge_vector=V_st,
            cutoff=select_cutoff,
        )

        (   
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(
            edge_index, cell_offsets, neighbors, D_st, V_st
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms
        )

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()
        pos.requires_grad_(True)
        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data)
        idx_s, idx_t = edge_index                                                                                                                                                                                                                                                                                    
        batch_per_atom = {
            atom: batch[atomic_numbers == atom] 
            for atom in self.atomic_numbers
        }
        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        res = self.descriptor(atomic_numbers,edge_index,D_st,id3_ba,id3_ca,cosφ_cab)
        if self.pca is not None:
            for atom in res.keys():
                a,w = self.scale[atom]
                res[atom] = (res[atom] -a.reshape(1,-1).to(edge_index.device))/w.reshape(1,-1).to(edge_index.device)
                res[atom] = torch.einsum('ij,jm->im', res[atom], self.pca[atom][0].to(edge_index.device))- self.pca[atom][2].reshape(1,-1).to(edge_index.device)
                res[atom] /= self.pca[atom][1].view(1,-1).to(edge_index.device)
        out = self.per_atom_fcn(res)        
        energy = torch.zeros(batch.max().item()+1, device=D_st.device)       
        for atom in out.keys():
            scatter(out[atom].squeeze(), batch_per_atom[atom], dim=0, out=energy)

        if self.regress_forces:
            forces = -torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _calculate_scale_and_fit_pca(self, dataset, bs=1):
       
        
       
        if torch.cuda.is_available():
            device="cuda" 
        else:
            device="cpu"
        self.descriptor = self.descriptor.to(device)
        scale = {}
      

        # split datasets w.r.t rank
        rank=distutils.get_rank()
        world_size=distutils.get_world_size()

        n=len(dataset)
        
        idx_=[n//world_size*(i) for i in range(world_size)]
        for j in range(n%world_size):
            idx_[-1-j] +=n%world_size-j 
        idx_.append(n)
        sz = (idx_[rank+1]-idx_[rank])//bs +2
        idxs = [bs*i + idx_[rank] if (bs*i + idx_[rank]<=idx_[rank+1] ) else idx_[rank+1] for i in np.arange(sz)]


        #compute mean(mu) of descriptors
        mu={}
        n_atoms={}
        with torch.no_grad():
            for i in range(sz-1):
                data_list = [dataset[j] for j in np.arange(idxs[i],idxs[i+1])]
                data = Batch.from_data_list(data_list)
                n_neighbors = []
                for i, data_ in enumerate(data_list):
                    n_index = data_.edge_index[1, :]
                    n_neighbors.append(n_index.shape[0])
                data.neighbors = torch.tensor(n_neighbors)
                data=data.to(device)

                atomic_numbers = data.atomic_numbers.long()
                (
                    edge_index,
                    neighbors,
                    D_st,
                    V_st,
                    id3_ba,
                    id3_ca,
                    id3_ragged_idx,
                ) = self.generate_interaction_graph(data)
                idx_s, idx_t = edge_index                                                                                                                                                                                                                                                                                    
                cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])

                descriptor = self.descriptor(atomic_numbers,edge_index,D_st,id3_ba,id3_ca,cosφ_cab)
                for atom in descriptor.keys():
                    if atom in mu.keys():
                        n_=descriptor[atom].shape[0]
                        theta=(n_atoms[atom])/(n_atoms[atom]+n_)
                        n_atoms[atom]+=n_
                        mu[atom]= mu[atom]*theta+descriptor[atom].mean(axis=0)*(1-theta)
                    else:
                        mu[atom]=descriptor[atom].mean(axis=0)
                        n_atoms[atom]=descriptor[atom].shape[0]

        #gather mu 
        for atom in mu.keys():
            n=n_atoms[atom]
            n_tot=distutils.all_reduce(torch.tensor(n).to(device))
            mu[atom] *=n_atoms[atom]/n_tot
            mu[atom]=distutils.all_reduce(mu[atom])
     
        
        
        
        # scale
        # we set scale mean=0,std=1 because it doesn't affect pca.
        for atom in mu.keys():
            scale[atom] = [torch.tensor(0.0),torch.tensor(1.0)]

        # pca
        
        # compute covariance matrix by batch
        XtX={}
        for atom in mu.keys():
            XtX[atom]=torch.zeros((self.descriptor.dim_descriptor,self.descriptor.dim_descriptor)).to(device)
        
        with torch.no_grad():
            for i in range(sz-1):
                data_list = [dataset[j] for j in np.arange(idxs[i],idxs[i+1])]
                data = Batch.from_data_list(data_list)
                n_neighbors = []
                for i, data_ in enumerate(data_list):
                    n_index = data_.edge_index[1, :]
                    n_neighbors.append(n_index.shape[0])
                data.neighbors = torch.tensor(n_neighbors)
                data=data.to(device)

                atomic_numbers = data.atomic_numbers.long()
                (
                    edge_index,
                    neighbors,
                    D_st,
                    V_st,
                    id3_ba,
                    id3_ca,
                    id3_ragged_idx,
                ) = self.generate_interaction_graph(data)
                idx_s, idx_t = edge_index                                                                                                                                                                                                                                                                                    
                cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])

                descriptor = self.descriptor(atomic_numbers,edge_index,D_st,id3_ba,id3_ca,cosφ_cab)

                for atom in descriptor.keys():
                    d=descriptor[atom]-mu[atom].reshape(1,-1)
                    XtX[atom]+=((d.unsqueeze(1))*(d.unsqueeze(-1))).sum(axis=0)

        #all-reduce covariance matrix
        for atom in XtX.keys():
            XtX[atom]=distutils.all_reduce(XtX[atom])
            n_atoms[atom]=distutils.all_reduce(torch.tensor(n_atoms[atom]).to(device))
        
        
       
        
        #calculate pca
        pca={}
        
        for atom in self.atomic_numbers:
            a,b=eigh(XtX[atom].detach().cpu().numpy())
            c=np.flip(b,axis=1).T
            max_abs_rows = np.argmax(np.abs(c), axis=1)
            signs = np.sign(c[range(c.shape[0]), max_abs_rows])
            c *= signs
            c=c.copy()
            sigma_=np.flip(a).copy()/(n_atoms[atom].detach().cpu().numpy()-1) 
            # eigh can result negative eigen value therefore set threshold for lower bound.
            idx_=np.where(sigma_<1e-8)
            sigma_[idx_]=1e-8
            pca[atom]=[torch.tensor(c.T),torch.tensor(np.sqrt(sigma_)),torch.tensor(np.dot(mu[atom].detach().cpu().numpy(), c.T))]

        return scale, pca

