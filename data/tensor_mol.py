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
    max_bonds = None
    grid_dim = None
    grid_coords = None
    grid_size = None
    grid_shape = None
    max_valence = None

    def set_cfg(cfg):
        TMCfg.max_atoms = cfg.max_atoms
        TMCfg.max_bonds = int(cfg.max_atoms*(cfg.max_atoms-1)*0.5)
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

    max_atoms = None
    max_bonds = None

    def get_bond_idx(start, end):
        return start*(TMCfg.max_atoms-1) - math.floor(start*(start+1)*0.5) + end - 1
        
    def __init__(self, mol=None, data=None):
        if mol is None:
            self.data = data
            return
        # batch, bond_idx, bond_type
        self.data = torch.zeros((TMCfg.max_bonds, NUM_BOND_TYPES))
        self.data[:,BOND_TYPE_HASH['_']] = 1 

    def add_bond(self, bond):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        idx = TensorBonds.get_bond_idx(start, end)
        self.data[idx,BOND_TYPE_HASH[bond_type]] = 1
        self.data[idx,BOND_TYPE_HASH['_']] = 0

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
        self.atom_types = torch.zeros((TMCfg.max_atoms, NUM_ATOM_TYPES))
        # batch, atom_idx, atom_valence
        self.atom_valences = torch.zeros((TMCfg.max_atoms, TMCfg.max_valence+1))

        for i, atom in enumerate(mol.GetAtoms()):
            atom_type = ATOM_TYPE_HASH[atom.GetSymbol()]
            self.atom_types[i,atom_type] = 1
            pos = mol.GetConformer().GetAtomPosition(i)
            self.coords[i] = torch.tensor([ pos.x, pos.y, pos.z ])
        self.center_coords()

        valences = [ 0 for i in range(TMCfg.max_atoms) ]
        self.bonds = TensorBonds(mol)
        for bond in mol.GetBonds():
            self.bonds.add_bond(bond)
            end_idx = bond.GetEndAtomIdx()
            valences[end_idx] += 1
        for i, valence in enumerate(valences):
            self.atom_valences[i,valence] = 1

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
            atom = torch.argmax(atom)
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
        
if __name__ == "__main__":
    test_collate()
