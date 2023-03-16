"""
written by robert.cho and byunggook.na (SAIT)
"""

import os
import numpy as np

# from ase.atom import atomic_numbers
import ase
from sklearn.decomposition import PCA

import torch
from torch.nn import Sequential, ModuleDict
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.data import Batch

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

        self.atom_types=atom_types
        self.module_dict={}
        for key in atom_types:
            module_list=[]
            for j in range(nLayer):
                if(j==0):
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

            self.module_dict[str(key)]= Sequential(*module_list)
        self.module_dict=ModuleDict(self.module_dict)
    def forward(self,x):

        out={}
        for key in self.atom_types:
            out[key]=self.module_dict[str(key)](x[key])
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
        self.cutoff=cutoff
        self.atomic_numbers=atomic_numbers

        if g2_params is None:
            g2_etas = torch.tensor([0.003214,0.035711,0.071421,0.124987,0.214264,0.357106,0.714213,1.428426])
        else:
            f = open(g2_params,'r')
            lines = f.readlines()
            g2_etas = torch.tensor([eval(v) for v in lines[0].split(",")])
            f.close()

        if g4_params is None:
            etas = torch.tensor([0.000357,0.028569,0.089277])
            zetas = torch.tensor([1.0,2.0,4.0])
            lmdas = torch.tensor([-1.0,1.0])
        else:
            f = open(g4_params,'r')
            lines = f.readlines()
            etas = torch.tensor([eval(v) for v in lines[0].split(",")])
            zetas = torch.tensor([eval(v) for v in lines[1].split(",")])
            lmdas = torch.tensor([eval(v) for v in lines[2].split(",")])
            f.close()

        n_species=len(self.atomic_numbers)
        self.dim_descriptor = n_species*len(g2_etas)+n_species*(n_species+1)*len(etas)*len(zetas)*len(lmdas)//2
        self.G2_params = {}
        self.G4_params_etas = {}
        self.G4_params_zetas = {}
        self.G4_params_lmdas = {}
        for atom1 in (self.atomic_numbers):
            for i,atom2 in enumerate(self.atomic_numbers):
                self.G2_params[str((atom1,atom2))]=torch.nn.Parameter(g2_etas.clone().detach(),            requires_grad=trainable)
                for j in range(i,len(self.atomic_numbers)):
                    atom3=self.atomic_numbers[j]
                    self.G4_params_etas[str((atom1,atom2,atom3))]=torch.nn.Parameter(etas.clone().detach(),requires_grad=trainable)
                    self.G4_params_zetas[str((atom1,atom2,atom3))]=torch.nn.Parameter(zetas.clone().detach(),requires_grad=trainable)
                    self.G4_params_lmdas[str((atom1,atom2,atom3))]=torch.nn.Parameter(lmdas.clone().detach(),requires_grad=trainable)
        self.G2_params      =torch.nn.ParameterDict(self.G2_params)
        self.G4_params_etas =torch.nn.ParameterDict(self.G4_params_etas)
        self.G4_params_zetas=torch.nn.ParameterDict(self.G4_params_zetas)
        self.G4_params_lmdas=torch.nn.ParameterDict(self.G4_params_lmdas)

    def forward(self,atomic_numbers ,edge_index,D_st,id3_ba,id3_ca,cosφ_cab):
        eba_idx= id3_ba[id3_ba>id3_ca]
        eca_idx= id3_ca[id3_ba>id3_ca]
        cos_idx= cosφ_cab[id3_ba>id3_ca]

        D_ba=D_st[eba_idx]
        D_ca=D_st[eca_idx]
        res_G4={}
        res_G2={}

        for atom in self.atomic_numbers:
            res_G4[atom]=[]
            res_G2[atom]=[]
        for atoms,eta in self.G4_params_etas.items():
            atom_a, atom_b,atom_c=eval(atoms)
            zta=self.G4_params_zetas[atoms]
            lmda=self.G4_params_lmdas[atoms]
            idx=((atomic_numbers[edge_index[:,eba_idx][0,:]]==atom_b)*(atomic_numbers[edge_index[:,eba_idx][1,:]]==atom_a)*(atomic_numbers[edge_index[:,eca_idx][0,:]]==atom_c))  \
                +((atomic_numbers[edge_index[:,eba_idx][0,:]]==atom_c)*(atomic_numbers[edge_index[:,eba_idx][1,:]]==atom_a)*(atomic_numbers[edge_index[:,eca_idx][0,:]]==atom_b)) 
            R_ba=D_ba[idx]
            R_ca=D_ca[idx]
            rcut=self.cutoff

            #Cos 제 2 법칙.       
            R_bc=(R_ba*R_ba + R_ca*R_ca-2*R_ba*R_ca*cos_idx[idx]).sqrt()
            cutoff=(0.5*torch.cos(torch.pi* R_ca/rcut)+0.5)*(0.5*torch.cos(torch.pi* R_bc/rcut)+0.5)*(0.5*torch.cos(torch.pi* R_ba/rcut)+0.5)*(R_bc<rcut)*(R_ba<rcut)*(R_ca<rcut)

            rad = torch.exp(-eta.reshape(1,-1)*(R_bc*R_bc + R_ba*R_ba + R_ca*R_ca).reshape(-1,1) )
            cos_lmda= torch.pow(torch.abs(1+lmda.reshape(1,-1,1)*cos_idx[idx].reshape(-1,1,1)),zta.reshape(1,1,-1))
            res=cos_lmda.reshape(cos_lmda.shape[0],1,cos_lmda.shape[1],cos_lmda.shape[2] )*rad.reshape(rad.shape+(1,1))*cutoff .reshape(-1,1,1,1)
            g4_per_atom=torch.zeros( (atomic_numbers.shape[0],)+res.shape[1:] ,device=res.device)
            scatter(res,edge_index[:,eba_idx[idx]][1,:],dim=0,out=g4_per_atom)
            res_G4[atom_a].append( torch.pow(2,1-zta)*  g4_per_atom[atomic_numbers==atom_a])

        for atoms,etas in self.G2_params.items():

            source_atom, target_atom=eval(atoms)

            idx_=(atomic_numbers[edge_index[ 1,:]]==target_atom)* (atomic_numbers[edge_index[ 0,:]]==source_atom)
            cutoff=0.5*torch.cos(torch.pi* D_st[idx_]/self.cutoff)+0.5
            etas=etas.to(D_st.device)
            rad = torch.exp(-(etas).reshape(1,-1) * (D_st[idx_]*D_st[idx_]).reshape(-1,1))
            res=cutoff.reshape(-1,1)*rad
            out_per_atom=torch.zeros( atomic_numbers.shape[0], len(etas),device=rad.device)
            scatter(res,edge_index[1,idx_],dim=0,out=out_per_atom)
            res_G2[target_atom].append(out_per_atom[atomic_numbers==target_atom])

        for key, val in res_G4.items():
            val=torch.stack(val,dim=-1)
            val=val.reshape(val.shape[0],-1)
            res_G4[key]=val
        for key, val in res_G2.items():
            val=torch.stack(val,dim=-1)
            val=val.reshape(val.shape[0],-1)
            res_G2[key]=val
        res={}
        for key in res_G2.keys():
            res[key]=torch.cat([res_G2[key],res_G4[key]],dim=-1)
        return res


# some functions are defined in GemNet (ocpmodels.models.gemnet.gemnet.py)
@registry.register_model("bpnn")
class BPNN(BaseModel):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        cutoff=5.0,
        max_neighbors=50,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        direct_forces=False,
        # BPNN arguments
        atom_species=None,
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
        assert self.cutoff <= 6 or otf_graph

        self.max_neighbors = max_neighbors
        #assert self.max_neighbors == 50 or otf_graph

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
        self.dim_descriptor=self.descriptor.dim_descriptor

        self.PerAtomspeciesNN = PerAtomFCN(
            nDescriptor=self.dim_descriptor,
            nHidden=nHidden,
            nLayer=nLayer,
            atom_types=self.atomic_numbers,
        )

        if dataset_path is None:
            self.scale = None
            self.pca = None
        else:
            config={}
            config['src']=dataset_path
            dset = LmdbDataset(config)
            
            if os.path.isfile(dataset_path):
                # single lmdb file
                scale_path = dataset_path.replace(".lmdb", "_scale_BPNN.pt")
                pca_path = dataset_path.replace(".lmdb", "_pca_BPNN.pt")
            elif os.path.isdir(dataset_path):
                # multi lmdb files
                # TODO: handling multi lmdb files
                scale_path = os.path.join(dataset_path, "scale_BPNN.pt")
                pca_path = os.path.join(dataset_path, "pca_BPNN.pt")
                raise NotImplementedError("Not implemented for dealing with multiple LMDB files.")

            # scale
            if os.path.exists(scale_path):
                scale = torch.load(scale_path)
                bm_logging.info(f"scale loaded from {scale_path}")
            else:
                scale = self._calculate_scale(dset, 32)
                torch.save(scale, scale_path)
                bm_logging.info(f"scale calculated and saved at {scale_path}")
            self.scale = scale

            # pca
            if os.path.exists(pca_path):
                pca = torch.load(pca_path)
                bm_logging.info(f"pca loaded from {pca_path}")
            else:
                pca = self._calculate_pca(dset, scale)
                torch.save(pca, pca_path)
                bm_logging.info(f"pca calculated and saved at {pca_path}")
            self.pca = pca

            for k, v in self.pca.items():
                self.pca[k] = [v[0], v[1].type(torch.FloatTensor), v[2]]

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
        batch_per_atom={}
        for atom in self.atomic_numbers:
            batch_per_atom[atom]= batch[atomic_numbers==atom]
        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        res=self.descriptor(atomic_numbers,edge_index,D_st,id3_ba,id3_ca,cosφ_cab)

        if(self.pca!=None):
            for atom in self.atomic_numbers:
                a,w=self.scale[atom]
                res[atom]=(res[atom] -a.reshape(1,-1).to(edge_index.device))/w.reshape(1,-1).to(edge_index.device)
                res[atom] = torch.einsum('ij,jm->im', res[atom], self.pca[atom][0].to(edge_index.device))- self.pca[atom][2].reshape(1,-1).to(edge_index.device)
                res[atom] /= self.pca[atom][1].view(1,-1).to(edge_index.device)
        out=self.PerAtomspeciesNN(res)        
        energy=torch.zeros(batch.max().item()+1,device=D_st.device)       
        for atom in self.atomic_numbers:
            scatter(out[atom].squeeze(),batch_per_atom[atom],dim=0,out=energy)

        force = -torch.autograd.grad(
                        energy.sum(), pos, create_graph=True
                    )[0]

        return energy, force

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _calculate_scale(self, dataset, bs=1):
        if torch.cuda.is_available():
            device="cuda" 
        else:
            device="cpu"
        self.descriptor=self.descriptor.to(device)
        scale={}
        sz=(len(dataset)-1)//bs +2
        idxs=[bs*i if bs*i<=len(dataset) else len(dataset) for i in np.arange(sz)]
        with torch.no_grad():
            for i in range(sz-1):
                data_list=[dataset[j] for j in np.arange(idxs[i],idxs[i+1])]
                data=Batch.from_data_list(data_list)
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
                for atom in self.atomic_numbers:
                    if( atom in scale.keys()):
                        scale[atom][:,0]=torch.minimum(descriptor[atom].min(axis=0).values,scale[atom][:,0])
                        scale[atom][:,1]=torch.maximum(descriptor[atom].max(axis=0).values,scale[atom][:,1])
                    else:
                        scale[atom]=torch.stack([descriptor[atom].min(axis=0).values,descriptor[atom].max(axis=0).values],axis=-1)
        for atom in self.atomic_numbers:
            a = (scale[atom][:,1]+scale[atom][:,0])/2
            w = (scale[atom][:,1]-scale[atom][:,0])/2
            scale[atom] = [a.cpu(),w.cpu()]
        return scale

    def _calculate_pca(self, dataset, scale, bs=32):
        pca={}
        if torch.cuda.is_available():
            device="cuda" 
        else:
            device="cpu"
        self.descriptor=self.descriptor.to(device)

        sz=(len(dataset)-1)//bs +2
        idxs=[bs*i if bs*i<=len(dataset) else len(dataset) for i in np.arange(sz)]
        descriptor_list={}
        for atom in self.atomic_numbers:
            descriptor_list[atom]=[]
        with torch.no_grad():
            for i in range(sz-1):
                data_list=[dataset[j] for j in np.arange(idxs[i],idxs[i+1])]
                data=Batch.from_data_list(data_list)
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
                cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])

                descriptor_=self.descriptor(atomic_numbers,edge_index,D_st,id3_ba,id3_ca,cosφ_cab)
                for atom in self.atomic_numbers:
                    descriptor_list[atom].append(descriptor_[atom].cpu())

        descriptor={}
        for atom in self.atomic_numbers:
            descriptor[atom] = torch.cat(descriptor_list[atom],dim=0)
        for atom in self.atomic_numbers:
                pca_temp = PCA()
                a, w = scale[atom]
                descriptor[atom] = (descriptor[atom]-a.reshape(1,-1))/w.reshape(1,-1)
                pca_temp.fit(descriptor[atom].detach().numpy())
                pca[atom] = [torch.tensor(pca_temp.components_.T),
                         torch.tensor(np.sqrt(pca_temp.explained_variance_ +1e-8)),
                         torch.tensor(np.dot(pca_temp.mean_, pca_temp.components_.T))]
        return pca