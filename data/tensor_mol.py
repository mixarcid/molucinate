from typing import List, Optional

import torch
from rdkit import Chem
import math
import numpy as np

try:
    from .chem import *
    from .utils import *
    from .dataloader import Collatable
except ImportError:
    from chem import *
    from utils import *
    from dataloader import Collatable

class TMCfg:

    max_atoms = None
    grid_dim = None
    grid_coords = None
    grid_size = None
    grid_shape = None
    max_valence = None

    def set_cfg(cfg):
        TMCfg.max_atoms = cfg.max_atoms
        TMCfg.grid_dim = cfg.grid_dim
        to_grid = []
        for i in range(3):
            to_grid.append(np.arange(-cfg.grid_dim/2 + cfg.grid_step/2,
                                     cfg.grid_dim/2 + cfg.grid_step/2,
                                     cfg.grid_step))
        grids = np.meshgrid(*to_grid)
        TMCfg.grid_coords = np.stack(grids, -1)
        TMCfg.grid_size = int(cfg.grid_dim/cfg.grid_step)
        TMCfg.grid_shape = (TMCfg.grid_size, TMCfg.grid_size, TMCfg.grid_size)
        TMCfg.max_valence = cfg.max_valence
    

class TensorBonds(Collatable):
    data: torch.Tensor
    all_indexes: Optional[List[List[int]]] = None
        
    def __init__(self, mol=None, data=None):
        if mol is None:
            self.data = data
            return
        # bond type, atom 1 idx, atom 2 idx
        self.data = torch.zeros((NUM_BOND_TYPES, TMCfg.max_atoms, TMCfg.max_atoms))
        self.data[BOND_TYPE_HASH['_']] = 1

    def add_bond(self, bond):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        for s, e in [(start, end), (end, start)]:
            self.data[BOND_TYPE_HASH[bond_type], s, e] = 1
            self.data[BOND_TYPE_HASH['_'], s, e] = 0

    def get_bond(self, start, end):
        if len(self.data.shape) == 3:
            return self.data[:,start,end]
        elif len(self.data.shape) == 4:
            return self.data[:,:,start,end]
        else:
            raise Exception("incorrect number of dims in TensorBonds data")

    def get_bond_argmax(self, start, end):
        if len(self.data.shape) == 3:
            return torch.argmax(self.data[:,start,end], 0)
        elif len(self.data.shape) == 4:
            return torch.argmax(self.data[:,:,start,end], 1)

    def get_all_indexes(self):
        if self.all_indexes is not None:
            return self.all_indexes
        out = []
        for end in range(TMCfg.max_atoms):
            for start in range(end):
                bonds = self.get_bond_argmax(start, end)
                if len(bonds.shape) == 0:
                    if bonds != BOND_TYPE_HASH['_']:
                        out.append([start, end, bonds])
                else:
                    for batch, bond in enumerate(bonds):
                        if bond != BOND_TYPE_HASH['_']:
                            out.append([batch, start, end, bond])
        self.all_indexes = out
        return out

    def argmax(self, atom_valences):
        raise NotImplementedError
                

# just stores atom coords, atom types, and bonds
# not used by any model directly, but useful intermediate
# also format used by renderer
class TensorMolBasic:
    atom_types: torch.Tensor
    atom_valences: torch.Tensor
    coords: torch.Tensor
    bonds: TensorBonds
        
    def __init__(self, mol):
        # batch, atom_idx, dimension
        self.coords = torch.zeros((TMCfg.max_atoms, 3))
        # batch, atom_idx, atom_type
        self.atom_types = torch.zeros(TMCfg.max_atoms, dtype=torch.long)
        # batch, atom_idx, atom_valence
        self.atom_valences = torch.zeros((TMCfg.max_atoms, NUM_ACT_BOND_TYPES), dtype=torch.long)

        for i, atom in enumerate(mol.GetAtoms()):
            atom_type = ATOM_TYPE_HASH[atom.GetSymbol()]
            self.atom_types[i] = atom_type
            pos = mol.GetConformer().GetAtomPosition(i)
            self.coords[i] = torch.tensor([ pos.x, pos.y, pos.z ])
        self.center_coords()

        self.bonds = TensorBonds(mol)
        for bond in mol.GetBonds():
            self.bonds.add_bond(bond)
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_index = BOND_TYPE_HASH[bond.GetBondType()]-1
            self.atom_valences[start_idx][bond_index] += 1
            self.atom_valences[end_idx][bond_index] += 1
            

    def center_coords(self):
        min_coord = torch.zeros((3,))
        max_coord = torch.zeros((3,))
        for i in range(3):
            min_coord[i] = min(self.coords[:,i])
            max_coord[i] = max(self.coords[:,i])
        center_coord = 0.5*(min_coord + max_coord)
        box_dims = max_coord - min_coord
        if torch.any(box_dims > TMCfg.grid_dim):
            raise Exception(f"molecule is too big! box_dims={box_dims}")
        self.coords -= center_coord


class TensorMol(Collatable):
    molgrid: torch.Tensor
    kps: torch.Tensor
    kps_1h: torch.Tensor
    atom_types: torch.Tensor
    atom_valences: torch.Tensor
    bonds: torch.Tensor

    def __init__(self,
                 mol=None,
                 molgrid=None,
                 kps=None,
                 kps_1h=None,
                 atom_types=None,
                 atom_valences=None,
                 bonds=None):
        if mol is None:
            self.molgrid = molgrid
            self.kps = kps
            self.kps_1h = kps_1h
            self.atom_types = atom_types
            self.atom_valences = atom_valences
            self.bonds = bonds
            return
        tmb = TensorMolBasic(mol)
        self.atom_types = tmb.atom_types
        self.atom_valences = tmb.atom_valences
        self.bonds = tmb.bonds

        sz = TMCfg.grid_size
        # batch, atom_idx, width, height, depth
        kp_shape = (TMCfg.max_atoms, sz, sz, sz)
        self.kps = torch.zeros(kp_shape)
        self.kps_1h = torch.zeros(kp_shape)
        # batch, atom_type, width, height, depth
        self.molgrid = torch.zeros((NUM_ATOM_TYPES, sz, sz, sz))

        for i, (coord, atom) in enumerate(zip(tmb.coords, self.atom_types)):
            if atom == ATOM_TYPE_HASH['_']: break
            grid = self.gridify_atom(coord, atom)
            self.kps[i] = grid
            self.kps_1h[i] = self.gridify_atom(coord, atom, True)
            self.molgrid[atom] += grid

    def gridify_atom(self, coord, atom, should_1h=False):
        #compute distances to each atom
        dists = np.linalg.norm(TMCfg.grid_coords - coord.cpu().numpy(), axis=-1)
        if should_1h:
            index = np.unravel_index(np.argmin(dists), dists.shape)
            A = np.zeros(TMCfg.grid_shape)
            A[index] = 1
        else:
            d2 = (dists**2)
            r = ATOM_RADII_LIST[atom]
            A0 = np.exp(-2*(d2/(r**2)))
            A1 = (4/((np.e**2)*(r**2)))*d2 - (12/((np.e**2)*r))*dists + 9/(np.e**2)
            A0_mask = dists < r
            A1_mask = np.logical_and(r <= dists, dists < 1.5*r)
            A = A0*A0_mask + A1*A1_mask
        return torch.tensor(A)

    def argmax(self):
        if self.atom_types.dtype != torch.long:
            if self.atom_valences is None:
                atom_valences = None
            else:
                atom_valences = torch.argmax(self.atom_valences, -1)
            atom_types = torch.argmax(self.atom_types, -1)
            # todo: re-add
            #bonds = self.bonds.argmax(self.bonds, atom_valences)
            bonds = self.bonds
            return TensorMol(
                None,
                self.molgrid,
                self.kps,
                self.kps_1h,
                atom_types,
                atom_valences,
                bonds
            )
        else:
            return self

    def get_coords(self):
        coords = []
        kps = self.kps if self.kps_1h is None else self.kps_1h
        for kp in kps:
            index = np.unravel_index(torch.argmax(kp).cpu().numpy(), kp.shape)
            coords.append(TMCfg.grid_coords[index[0], index[1], index[2]])
        return torch.tensor(coords)

    def atom_str(self):
        ret = ''
        for atom in self.atom_types:
            ret += ATOM_TYPE_LIST[atom]
        return ret


def test_basic():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'grid_dim': 16,
        'grid_step': 0.5,
        'max_atoms': 38,
        'max_valence': 6
    })
    TMCfg.set_cfg(cfg)
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(mol)
    print(torch.amax(tm.molgrid))
    print(tm.atom_valences)

def test_collate():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'grid_dim': 16,
        'grid_step': 0.5,
        'max_atoms': 38,
        'max_valence': 6
    })
    TMCfg.set_cfg(cfg)
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    tm = TensorMol(mol)
    for attr in tm.recurse():
        print(attr)

def test_bond_argmax():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'grid_dim': 16,
        'grid_step': 0.5,
        'max_atoms': 38,
        'max_valence': 6
    })
    TMCfg.set_cfg(cfg)
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    tm = TensorMol(mol)
    bdata = torch.randn((NUM_BOND_TYPES, TMCfg.max_atoms, TMCfg.max_atoms))
    bonds = TensorBonds(data=bdata)
    print(bonds.argmax(tm.atom_valences))
    
if __name__ == "__main__":
    test_bond_argmax()
