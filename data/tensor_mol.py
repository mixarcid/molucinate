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


class TensorBonds(Collatable):
    data: torch.Tensor
    max_atoms: int

    def __init__(self, cfg=None, mol=None, data=None, example=None, collate_len=None):
        super().__init__(example, collate_len)
        if cfg is None:
            self.data = data
            if example is not None:
                self.max_atoms = example.max_atoms
            return
        self.max_atoms = cfg.max_atoms
        # batch, bond_idx, bond_type
        self.data = torch.zeros((self.get_max_bonds(), NUM_BOND_TYPES))
        self.data[:,BOND_TYPE_HASH['_']] = 1

    def get_max_bonds(self):
        return int(self.max_atoms*(self.max_atoms-1)*0.5)

    def get_bond_idx(self, start, end):
        return start*(self.max_atoms-1) - math.floor(start*(start+1)*0.5) + end - 1

    def add_bond(self, bond):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        idx = self.get_bond_idx(start, end)
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

    def __init__(self, cfg, mol):
        # batch, atom_idx, dimension
        self.coords = torch.zeros((cfg.max_atoms, 3))
        # batch, atom_idx, atom_type
        self.atom_types = torch.zeros((cfg.max_atoms, NUM_ATOM_TYPES))
        # batch, atom_idx, atom_valence
        self.atom_valences = torch.zeros((cfg.max_atoms, cfg.max_valence+1))

        for i, atom in enumerate(mol.GetAtoms()):
            atom_type = ATOM_TYPE_HASH[atom.GetSymbol()]
            self.atom_types[i,atom_type] = 1
            pos = mol.GetConformer().GetAtomPosition(i)
            self.coords[i] = torch.tensor([ pos.x, pos.y, pos.z ])
        self.center_coords(cfg)

        valences = [ 0 for i in range(cfg.max_atoms) ]
        self.bonds = TensorBonds(cfg, mol)
        for bond in mol.GetBonds():
            self.bonds.add_bond(bond)
            end_idx = bond.GetEndAtomIdx()
            valences[end_idx] += 1
        for i, valence in enumerate(valences):
            self.atom_valences[i,valence] = 1

    def center_coords(self, cfg):
        min_coord = torch.zeros((3,))
        max_coord = torch.zeros((3,))
        for i in range(3):
            min_coord[i] = min(self.coords[:,i])
            max_coord[i] = max(self.coords[:,i])
        center_coord = 0.5*(min_coord + max_coord)
        box_dims = max_coord - min_coord
        if torch.any(box_dims > cfg.grid_dim):
            raise Exception(f"molecule is too big! box_dims={box_dims}")
        self.coords -= center_coord


class TensorMol(Collatable):
    molgrid: torch.Tensor
    kps: torch.Tensor
    kps_1h: torch.Tensor
    atom_types: torch.Tensor
    atom_valences: torch.Tensor
    bonds: torch.Tensor

    grid_coords = None

    def __init__(self,
                 cfg=None,
                 mol=None,
                 molgrid=None,
                 kps=None,
                 kps_1h=None,
                 atom_types=None,
                 atom_valences=None,
                 bonds=None,
                 example=None,
                 collate_len=None):
        super().__init__(example, collate_len)
        if cfg is None:
            self.molgrid = molgrid
            self.kps = kps
            self.kps_1h = kps_1h
            self.atom_types = atom_types
            self.atom_valences = atom_valences
            self.bonds = bonds
            return
        TensorMol.ensure_grid_coords(cfg)
        tmb = TensorMolBasic(cfg, mol)
        self.atom_types = tmb.atom_types
        self.atom_valences = tmb.atom_valences
        self.bonds = tmb.bonds

        sz = get_grid_size(cfg)
        self.grid_shape = (sz,sz,sz)
        self.grid_dim = cfg.grid_dim
        # batch, atom_idx, width, height, depth
        kp_shape = (cfg.max_atoms, sz, sz, sz)
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
            
    def ensure_grid_coords(cfg):
        if TensorMol.grid_coords is not None: return
        to_grid = []
        for i in range(3):
            to_grid.append(np.arange(-cfg.grid_dim/2 + cfg.grid_step/2,
                                     cfg.grid_dim/2 + cfg.grid_step/2,
                                     cfg.grid_step))
        grids = np.meshgrid(*to_grid)
        TensorMol.grid_coords = np.stack(grids, -1)


    def gridify_atom(self, coord, atom, should_1h=False):
        #compute distances to each atom
        dists = np.linalg.norm(TensorMol.grid_coords - coord.cpu().numpy(), axis=-1)
        if should_1h:
            index = np.unravel_index(np.argmin(dists), dists.shape)
            A = np.zeros(self.grid_shape)
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
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(cfg, mol)
    print(torch.amax(tm.molgrid))

def test_collate():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'grid_dim': 16,
        'grid_step': 0.5,
        'max_atoms': 38,
        'max_valence': 6
    })
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    tm = TensorMol(cfg, mol)
    for attr in tm.recurse():
        print(attr)
        
if __name__ == "__main__":
    test_collate()
