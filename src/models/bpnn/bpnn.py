"""
written by robert.cho and byunggook.na (SAIT)
"""

import os
import numpy as np
from pathlib import Path
from scipy.linalg import eigh

import ase

import torch
from torch.nn import Sequential, ModuleDict
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.data import Batch

from ocpmodels.datasets import LmdbDataset
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

        self.max_atom=torch.max(torch.tensor(self.atomic_numbers))+1
        atom_to_index=(self.max_atom+1)*torch.ones(self.max_atom,dtype=torch.long)


        self.register_buffer("atom_to_index", atom_to_index)

        atomic_number_str=[str(v) for v in atomic_numbers]
        atomic_number_str.sort()
        
        for i,atomic_number in enumerate(atomic_number_str):
            atomic_number=eval(atomic_number)
            self.atom_to_index[atomic_number]=i
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
        self.n_species =n_species
        self.g2_shape=(len(g2_etas),n_species)
        self.g4_shape=(len(etas),len(lmdas),len(zetas),(n_species*(n_species+1)) //2)
        self.dim_descriptor = n_species*len(g2_etas) + n_species*(n_species+1)*len(etas)*len(zetas)*len(lmdas)//2
        self.G2_params = torch.nn.Parameter(torch.zeros(       (n_species,n_species,len(g2_etas)) ),requires_grad=trainable)
        self.G4_params_etas = torch.nn.Parameter(torch.zeros(  (n_species,n_species,n_species,len(etas)) ),requires_grad=trainable)
        self.G4_params_zetas = torch.nn.Parameter(torch.zeros( (n_species,n_species,n_species,len(zetas)) ),requires_grad=trainable)
        self.G4_params_lmdas = torch.nn.Parameter(torch.zeros( (n_species,n_species,n_species,len(lmdas)) ),requires_grad=trainable)


        for i,atom1 in enumerate(self.atomic_numbers):
            for j, atom2 in enumerate(self.atomic_numbers):
                self.G2_params[i,j,:] = g2_etas.clone().detach()
                for k in range( len(self.atomic_numbers)):
                    atom3 = self.atomic_numbers[k]
                    self.G4_params_etas [i,j,k,:] =  etas.clone().detach()
                    self.G4_params_zetas[i,j,k,:] = zetas.clone().detach()
                    self.G4_params_lmdas[i,j,k,:] = lmdas.clone().detach()



        ## this is dummy.... need to be fixed later.... for index sorting

        G2_params=[]
        G4_params=[]
        for atom1 in self.atomic_numbers:
            for i, atom2 in enumerate(self.atomic_numbers):
                G2_params.append(str((atom1,atom2)) )
                for j in range(i, len(self.atomic_numbers)):
                    atom3 = self.atomic_numbers[j]
                    G4_params.append(str((atom1,atom2,atom3)))
        idx_mapping=torch.zeros(self.max_atom,self.max_atom,self.max_atom,dtype=torch.long)
        self.register_buffer("idx_mapping", idx_mapping)
        count=torch.zeros(self.max_atom,dtype=torch.long)
        G4_params.sort()
        G2_params.sort()
        for atoms_triplet in G4_params:

            atoms_triplet=eval(atoms_triplet)
            self.idx_mapping[atoms_triplet[0],atoms_triplet[2],atoms_triplet[1]]=count[atoms_triplet[0]]
            self.idx_mapping[atoms_triplet[0],atoms_triplet[1],atoms_triplet[2]]=count[atoms_triplet[0]]
            count[atoms_triplet[0]]+=1
        idx_mapping_g2=torch.zeros(self.max_atom,self.max_atom,dtype=torch.long)
        self.register_buffer("idx_mapping_g2", idx_mapping_g2)
        count=torch.zeros(self.max_atom,dtype=torch.long)
        for atoms_triplet in G2_params:
            atoms_triplet=eval(atoms_triplet)
            self.idx_mapping_g2[atoms_triplet[0],atoms_triplet[1]]=count[atoms_triplet[0]]
            count[atoms_triplet[0]]+=1
    def forward(self, atomic_numbers, edge_index, D_st, id3_ba, id3_ca, cosφ_cab):
        eba_idx = id3_ba[id3_ba>id3_ca]
        eca_idx = id3_ca[id3_ba>id3_ca]
        cos_idx = cosφ_cab[id3_ba>id3_ca]
        D_ba = D_st[eba_idx]
        D_ca = D_st[eca_idx]
        atom_exist = [atom  for atom in self.atomic_numbers if (atomic_numbers == atom).sum() > 0]
        atomb=atomic_numbers[edge_index[:,eba_idx][0,:]] 
        atomc=atomic_numbers[edge_index[:,eca_idx][0,:]] 
        atoma=atomic_numbers[edge_index[:,eca_idx][1,:]] 
        lmdas=self.G4_params_lmdas[self.atom_to_index[atoma],self.atom_to_index[atomb],self.atom_to_index[atomc],:]
        etas=self.G4_params_etas  [self.atom_to_index[atoma],self.atom_to_index[atomb],self.atom_to_index[atomc],:]
        zetas=self.G4_params_zetas[self.atom_to_index[atoma],self.atom_to_index[atomb],self.atom_to_index[atomc],:]
        R_ba = D_ba                                                                                                                                                                    
        R_ca = D_ca
        R_bc = (R_ba*R_ba + R_ca*R_ca-2*R_ba*R_ca*cos_idx).sqrt()
        cutoff = ((0.5*torch.cos(torch.pi*R_ca/self.cutoff) + 0.5) * 
                (0.5*torch.cos(torch.pi*R_bc/self.cutoff) + 0.5) * 
                        (0.5*torch.cos(torch.pi*R_ba/self.cutoff) + 0.5) * 
                        (R_bc<self.cutoff) * 
                        (R_ba<self.cutoff) * 
                        (R_ca<self.cutoff)
                    )
        rad = torch.exp(-etas*(R_bc*R_bc + R_ba*R_ba + R_ca*R_ca).reshape(-1,1))
        cos_lmda = torch.pow(torch.abs(1+lmdas.unsqueeze(2)*cos_idx.reshape(-1,1,1)), zetas.unsqueeze(1))
        res = cos_lmda.unsqueeze(1) * rad.reshape(rad.shape+(1,1)) * cutoff.reshape(-1,1,1,1)
        res=torch.pow(2, 1-zetas.unsqueeze(1)).unsqueeze(1)*res

        res_g4=torch.zeros(self.g4_shape+atomic_numbers.shape)
        res_g4=res_g4.reshape(-1,*res_g4.shape[:-2])


        descriptor_idx=self.idx_mapping[atoma,atomb,atomc]
        descriptor_idx=descriptor_idx.to(edge_index.device)
        res_g4=res_g4.to(edge_index.device)
        idx=edge_index[:,eca_idx][1,:]+descriptor_idx*len(atomic_numbers)
        scatter(res,idx,dim=0,out=res_g4)
        res_g4=res_g4.reshape((self.n_species*(self.n_species+1))//2,len(atomic_numbers),*res_g4.shape[1:])
        res_g4=res_g4.permute(1,2,3,4,0)


        cutoff = 0.5*torch.cos(torch.pi*D_st/self.cutoff) + 0.5
        eta=self.G2_params[self.atom_to_index[atomic_numbers[edge_index[1,:]]],self.atom_to_index[atomic_numbers[edge_index[0,:]]],:]
        eta = eta.to(D_st.device)

        rad = torch.exp(-(eta)* (D_st*D_st).reshape(-1,1))
        res = cutoff.reshape(-1,1) * rad


        res_g2 = torch.zeros(self.g2_shape+atomic_numbers.shape).to(edge_index.device)
        res_g2=res_g2.reshape(-1,*res_g2.shape[:-2])
        descriptor_idx=self.idx_mapping_g2[atomic_numbers[edge_index[1,:]],atomic_numbers[edge_index[0,:]]]
        descriptor_idx=descriptor_idx.to(edge_index.device)
        idx=edge_index[1,:]+descriptor_idx*len(atomic_numbers)
        scatter(res, idx, dim=0, out=res_g2)
        res_g2=res_g2.reshape(self.n_species ,len(atomic_numbers),*res_g2.shape[1:])
        res_g2=res_g2.permute(1,2,0)


        res={}
        for atom in atom_exist:
            res[atom]=res_g4[atom==atomic_numbers ,:]
            g2=res_g2[atom==atomic_numbers ,:]
            g4=res_g4[atom==atomic_numbers ,:]
            res[atom]=torch.cat((g2.reshape(g2.shape[0],-1),g4.reshape(g4.shape[0],-1)),dim=-1)
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
        use_pca=True, 
        pca_path=None,
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

        if not use_pca:
            bm_logging.warning("PCA is not used (if using PCA, BPNN accuarcy could be improved)")
            self.pca = None
        else:
            # set PCA
            if pca_path is None:
                pca_path = Path(dataset_path).parent / "BPNN_pca.pt"

            if os.path.exists(pca_path):
                pca = torch.load(pca_path)
                bm_logging.info(f"The fitted PCA is loaded from {pca_path}")
            else:
                assert dataset_path is not None
                bm_logging.info(f"Start PCA fitting ... ")
                pca = self._fit_pca(
                    dataset=LmdbDataset({'src': dataset_path}), 
                    bs=256,
                )
                torch.save(pca, pca_path)
                bm_logging.info(f"The fitted PCA is saved at {pca_path}")
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
        ) = self.generate_graph(data, max_neighbors=self.max_neighbors)
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
        device = edge_index.device
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        res = self.descriptor(atomic_numbers, edge_index, D_st, id3_ba, id3_ca, cosφ_cab)
        if self.pca is not None:
            for atom in res.keys():
                res[atom] = torch.einsum('ij,jm->im', res[atom], self.pca[atom][0].to(device))- self.pca[atom][2].reshape(1,-1).to(device)
                res[atom] /= self.pca[atom][1].view(1,-1).to(device)
        out = self.per_atom_fcn(res)
        energy = torch.zeros(batch.max().item()+1, device=device)
        for atom in out.keys():
            scatter(out[atom].reshape(-1), batch_per_atom[atom], dim=0, out=energy)
        if self.regress_forces:
            forces = -torch.autograd.grad(energy.sum(), pos, create_graph=True)[0]
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def _fit_pca(self, dataset, bs=32):
        if torch.cuda.is_available():
            device="cuda" 
        else:
            device="cpu"
        self.descriptor = self.descriptor.to(device)

        # split datasets w.r.t rank
        rank = distutils.get_rank()
        world_size = distutils.get_world_size()

        n = len(dataset)
        idx_ = [(n // world_size) * i for i in range(world_size)]
        for j in range(n % world_size):
            idx_[-1-j] += n % world_size - j
        idx_.append(n)
        sz = (idx_[rank+1] - idx_[rank])//bs + 2
        idxs = [bs*i + idx_[rank] if (bs*i + idx_[rank] <= idx_[rank+1] ) else idx_[rank+1] for i in range(sz)]

        # compute mean(mu) of descriptors
        mu = {}
        n_atoms = {}
        XtX = {}
        for atom in self.atomic_numbers:
            XtX[atom] = torch.zeros((self.descriptor.dim_descriptor, self.descriptor.dim_descriptor)).to(device)
            mu[atom] = torch.zeros(self.descriptor.dim_descriptor).to(device)
            n_atoms[atom] = torch.zeros(1, dtype=torch.int).to(device)
        
        with torch.no_grad():
            for i in range(sz-1):
                data_list = [dataset[j] for j in range(idxs[i], idxs[i+1])]
                data = Batch.from_data_list(data_list)
                # if not self.otf_graph:
                    # raise NotImplementedError("To fit PCA, edges are required. Please set otf_graph=False with a dataset including edges")
                n_neighbors = []
                for i, data_ in enumerate(data_list):
                    n_index = data_.edge_index[1, :]
                    n_neighbors.append(n_index.shape[0])
                data.neighbors = torch.tensor(n_neighbors)
                data = data.to(device)

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
                descriptor = self.descriptor(atomic_numbers, edge_index, D_st, id3_ba, id3_ca, cosφ_cab)

                for atom in descriptor.keys():
                    d = descriptor[atom]
                    n_ = descriptor[atom].shape[0]
                    theta = n_atoms[atom] / (n_atoms[atom] + n_)
                    XtX[atom] = theta*XtX[atom] + (1-theta)*(d.unsqueeze(1) * d.unsqueeze(-1)).mean(axis=0)
                    mu[atom] = theta*mu[atom] + (1-theta)*descriptor[atom].mean(axis=0)
                    n_atoms[atom] += n_
               
        # gather mu 
        for atom in mu.keys():
            n = n_atoms[atom]
            n_tot = distutils.all_reduce(torch.tensor(n).to(device))
            mu[atom] *= n_atoms[atom]/n_tot
            mu[atom] = distutils.all_reduce(mu[atom])

        # gather covariance matrix
        for atom in XtX.keys():
            n = n_atoms[atom]
            n_tot = distutils.all_reduce(torch.tensor(n).to(device))
            XtX[atom] *= n_atoms[atom]/n_tot
            XtX[atom] = distutils.all_reduce(XtX[atom])
        
        # substract muTmu
        for atom in XtX.keys():
            XtX[atom] = XtX[atom] - (mu[atom].unsqueeze(0) * mu[atom].unsqueeze(1))

        # calculate pca
        pca = {}
        for atom in self.atomic_numbers:
            a, b = eigh(XtX[atom].detach().cpu().numpy())
            c = np.flip(b,axis=1).T
            max_abs_rows = np.argmax(np.abs(c), axis=1)
            signs = np.sign(c[range(c.shape[0]), max_abs_rows])
            c *= signs.reshape(-1, 1)
            c = c.copy()
            sigma_ = (n_tot.detach().cpu().numpy() / (n_tot.detach().cpu().numpy() - 1)) * np.flip(a).copy()
            # eigh can result negative eigen value therefore set threshold for lower bound.
            idx_ = np.where(sigma_ < 1e-8)
            sigma_[idx_] = 1e-8
            pca[atom] = [
                torch.tensor(c.T),
                torch.tensor(np.sqrt(sigma_)),
                torch.tensor(np.dot(mu[atom].detach().cpu().numpy(), c.T))
            ]

        return pca