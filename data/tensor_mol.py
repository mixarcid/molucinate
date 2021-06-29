from rdkit import Chem
from rdkit.Geometry import Point3D

from typing import List, Optional

import torch
import math
import numpy as np
from copy import deepcopy

try:
    from .chem import *
    from .utils import *
    from .dataloader import Collatable, collate
except ImportError:
    from chem import *
    from utils import *
    from dataloader import Collatable, collate

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
    bond_types: torch.Tensor
    bonded_atoms: torch.Tensor

    def __init__(self, mol=None,
                 bond_types=None,
                 bonded_atoms=None,
                 device='cpu'):
        if mol is None:
            self.bond_types = bond_types
            self.bonded_atoms = bonded_atoms
            return

        self.bond_types = torch.zeros((TMCfg.max_atoms, TMCfg.max_valence), dtype=torch.long) #NUM_BOND_TYPES
        self.bonded_atoms = torch.zeros((TMCfg.max_atoms, TMCfg.max_valence), dtype=torch.long) #TMCfg.max_atoms

        atom_valences = torch.zeros((TMCfg.max_atoms), dtype=torch.long)
        
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            cur_valence = atom_valences[end_idx]
            self.bond_types[end_idx][cur_valence] = BOND_TYPE_HASH[bond.GetBondType()]
            self.bonded_atoms[end_idx][cur_valence] = start_idx
            atom_valences[end_idx] += 1

    def argmax(self, atom_types):
        assert self.bond_types.dtype == torch.float and self.bonded_atoms.dtype == torch.float
        bond_types = torch.argmax(self.bond_types, -1)
        bonded_atoms = torch.argmax(self.bonded_atoms, -1)
        return TensorBonds(None, bond_types, bonded_atoms)

    def get_all_indexes(self):
        assert self.bond_types.dtype == torch.long and self.bonded_atoms.dtype == torch.long
        out = []
        if self.bond_types.shape[0] == 0:
            return []
        if len(self.bond_types.shape) == 3:
            for batch in range(self.bond_types.size(0)):
                for end in range(1, self.bond_types.size(1)):
                    for bi in range(self.bond_types.size(2)):
                        bond = int(self.bond_types[batch][end][bi])
                        start = int(self.bonded_atoms[batch][end][bi])
                        if bond != BOND_TYPE_HASH['_']:
                            assert(start < end)
                            out.append([batch, start, end, bond])
        elif len(self.bond_types.shape) == 2:
            for end in range(1, self.bond_types.size(0)):
                for bi in range(self.bond_types.size(1)):
                    bond = int(self.bond_types[end][bi])
                    start = int(self.bonded_atoms[end][bi])
                    if bond != BOND_TYPE_HASH['_']:
                        assert(start < end)
                        out.append([start, end, bond])
        else:
            raise Exception(f"Incorrect number of dims in TensorBonds data {self.bond_types.shape=}")
        #poop
        return out

class TensorBondsValence(Collatable):
    data: torch.Tensor
    all_indexes: Optional[List[List[int]]] = None
        
    def __init__(self, mol=None, data=None, device='cpu'):
        if data is not None:
            self.data = data
            return
        self.data = self.get_default_data(device)

        atom_valences = torch.zeros((TMCfg.max_atoms, NUM_ACT_BOND_TYPES), dtype=torch.long)

        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_index = BOND_TYPE_HASH[bond.GetBondType()]-1
            cur_valence = atom_valences[end_idx][bond_index]
            self.data[end_idx][bond_index][cur_valence] = start_idx
            atom_valences[end_idx][bond_index] += 1

    def get_default_data(self, device='cpu', batches=None, atoms=None):
        if atoms is None:
            atoms = TMCfg.max_atoms
        if batches is None:
            shape = (atoms, NUM_ACT_BOND_TYPES, TMCfg.max_valence)
        else:
            shape = (batches, atoms, NUM_ACT_BOND_TYPES, TMCfg.max_valence)
        data = torch.zeros(shape, device=device, dtype=torch.long)
        for i in range(atoms):
            if batches is None:
                data[i,:,:] = i
            else:
                data[:,i,:,:] = i
        return data

    def argmax(self, atom_types):
        assert self.data.dtype == torch.float
        data = torch.argmax(self.data, -1)
        idxs = (atom_types != ATOM_TYPE_HASH["_"]) & (atom_types != ATOM_TYPE_HASH["^"])
        if len(data.shape) == 3:
            batches = None
            atoms = None
        else:
            batches = self.data.shape[0]
            atoms = self.data.shape[1]
        ret_data = self.get_default_data(self.data.device, batches, atoms)
        ret_data[idxs] = data[idxs]
        return TensorBondsValence(data=ret_data)

    def get_all_indexes(self):
        assert self.data.dtype == torch.long
        if self.data.shape[0] == 0: return []
        if len(self.data.shape) == 3:
            batches = None
            atoms = None
        else:
            batches = self.data.shape[0]
            atoms = self.data.shape[1]
        mask = self.data != self.get_default_data(self.data.device, batches, atoms)
        indexes = mask.nonzero(as_tuple=False)
        #print(self.data)
        #print(indexes)
        # shape is [batch, start, end, bond]
        end = deepcopy(indexes[...,-3])
        indexes[...,-1] = indexes[...,-2] + 1
        indexes[...,-2] = end
        indexes[...,-3] = self.data[mask]
        if len(indexes) == 0:
            return []
        indexes = set(indexes)
        return indexes

# just stores atom coords, atom types, and bonds
# not used by any model directly, but useful intermediate
# also format used by renderer
class TensorMolBasic:
    atom_types: torch.Tensor
    #atom_valences: torch.Tensor
    coords: torch.Tensor
    bonds: TensorBonds
        
    def __init__(self, mol):
        # batch, atom_idx, dimension
        self.coords = torch.zeros((TMCfg.max_atoms, 3))
        # batch, atom_idx, atom_type
        self.atom_types = torch.zeros(TMCfg.max_atoms, dtype=torch.long)
        # batch, atom_idx, atom_valence
        #self.atom_valences = torch.zeros((TMCfg.max_atoms, NUM_ACT_BOND_TYPES), dtype=torch.long)

        for i, atom in enumerate(mol.GetAtoms()):
            atom_type = ATOM_TYPE_HASH[atom.GetSymbol()]
            self.atom_types[i] = atom_type
            pos = mol.GetConformer().GetAtomPosition(i)
            self.coords[i] = torch.tensor([ pos.x, pos.y, pos.z ])
        self.center_coords()

        self.bonds = TensorBonds(mol)
        """for bond in mol.GetBonds():
            self.bonds.add_bond(bond)
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_index = BOND_TYPE_HASH[bond.GetBondType()]-1
            #self.atom_valences[start_idx][bond_index] += 1
            self.atom_valences[end_idx][bond_index] += 1"""
            

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
    #atom_valences: torch.Tensor
    bonds: TensorBonds

    def __init__(self,
                 mol=None,
                 molgrid=None,
                 kps=None,
                 kps_1h=None,
                 atom_types=None,
                 #atom_valences=None,
                 bonds=None):
        if mol is None:
            self.molgrid = molgrid
            self.kps = kps
            self.kps_1h = kps_1h
            self.atom_types = atom_types
            #self.atom_valences = atom_valences
            self.bonds = bonds
            return
        tmb = TensorMolBasic(mol)
        self.atom_types = tmb.atom_types
        #self.atom_valences = tmb.atom_valences
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
            #if self.atom_valences is None:
            #    atom_valences = None
            #else:
            #    atom_valences = torch.argmax(self.atom_valences, -1)
            atom_types = torch.argmax(self.atom_types, -1)
            bonds = self.bonds.argmax(atom_types)
            if self.kps is None and self.kps_1h is not None:
                kps = torch.zeros_like(self.kps_1h, device=self.kps_1h.device)
                if len(kps.shape) == 4:
                    coords = self.get_coords()
                    for i, (coord, atom) in enumerate(zip(coords, atom_types)):
                        if atom in [ ATOM_TYPE_HASH['_'], ATOM_TYPE_HASH['^'] ]: continue
                        grid = self.gridify_atom(coord, atom)
                        kps[i] = grid
                elif len(kps.shape) == 5:
                    for i in range(kps.size(0)):
                        coords = self[i].get_coords()
                        for j, (coord, atom) in enumerate(zip(coords, atom_types[i])):
                            if atom in [ ATOM_TYPE_HASH['_'], ATOM_TYPE_HASH['^'] ]: continue
                            grid = self.gridify_atom(coord, atom)
                            kps[i][j] = grid
                else:
                     raise Exception("Incorrect shape for kps_1h")   
            else:
                kps = self.kps
            return TensorMol(
                None,
                self.molgrid,
                kps,
                self.kps_1h,
                atom_types,
                #atom_valences,
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

    def get_mol(self, add_conformer=True):
        mol = Chem.RWMol()
        mol_idxs = []
        idx = 0
        for atom in self.atom_types:
            if atom not in [ATOM_TYPE_HASH['_'], ATOM_TYPE_HASH['^']]:
                mol.AddAtom(Chem.Atom(ATOM_TYPE_LIST[atom]))
                mol_idxs.append(idx)
                idx += 1
            else:
                mol_idxs.append(None)
        for start, end, bond in self.bonds.get_all_indexes():
            if mol_idxs[start] is None or mol_idxs[end] is None:
                continue
            try:
                mol.AddBond(mol_idxs[start], mol_idxs[end], BOND_TYPE_LIST[bond])
            except RuntimeError:
                pass
        coords = self.get_coords()
        conformer = Chem.Conformer(mol.GetNumAtoms())
        if add_conformer:
            for i, coord in enumerate(coords):
                if self.atom_types[i] not in [ATOM_TYPE_HASH['_'], ATOM_TYPE_HASH['^']]:
                    conformer.SetAtomPosition(mol_idxs[i], Point3D(float(coords[i][0]),
                                                                   float(coords[i][1]),
                                                                   float(coords[i][2])))
            mol.AddConformer(conformer)
        return mol

def get_test_mol():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'grid_dim': 16,
        'grid_step': 0.5,
        'max_atoms': 38,
        'max_valence': 6
    })
    TMCfg.set_cfg(cfg)
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    return mol

def test_basic():
    mol = get_test_mol()
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(mol)
    print(torch.amax(tm.molgrid))
    print(tm.atom_valences)

def test_collate():
    mol = get_test_mol()
    tm = TensorMol(mol)
    for attr in tm.recurse():
        print(attr)

def test_bond_argmax():
    mol = get_test_mol()
    tm = TensorMol(mol)
    bdata = torch.randn((TMCfg.max_atoms, NUM_ACT_BOND_TYPES, TMCfg.max_valence, TMCfg.max_atoms))
    for i in range(TMCfg.max_atoms):
        bdata[i,:,:,i:] = 0
    bonds = TensorBondsValence(data=bdata)
    bonds_a = bonds.argmax(tm.atom_types)
    print(bonds_a.get_all_indexes())

def test_mol_export():
    from rdkit.Chem import Draw
    mol = get_test_mol()
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(mol)
    mol2 = tm.get_mol()
    print(Chem.MolToSmiles(mol2))
    Draw.MolToFile(mol2, "test_output/mol_3d.png", size=(500, 500))
    Draw.MolToFile(tm.get_mol(False), "test_output/mol_2d.png", size=(500, 500))

def test_bond_recon():
    from rdkit.Chem import Draw
    mol = get_test_mol()
    Chem.Kekulize(mol)
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(mol)
    mol2 = tm.get_mol()
    #print(tm.bonds.data)
    print(Chem.MolToSmiles(mol2))
    Draw.MolToFile(mol, "test_output/mol_og.png", size=(500, 500), kekulize=False)
    Draw.MolToFile(mol2, "test_output/mol_3d.png", size=(500, 500), kekulize=False)
    Draw.MolToFile(tm.get_mol(False), "test_output/mol_2d.png", size=(500, 500), kekulize=False)

def test_collated_bond_indexes():
    from rdkit.Chem import Draw
    mol = get_test_mol()
    print(Chem.MolToSmiles(mol))
    tm = TensorMol(mol)
    tm_collated = collate([tm, tm])
    print(tm.bonds.get_all_indexes())
    print(tm_collated.bonds.get_all_indexes())
    
if __name__ == "__main__":
    #test_mol_export()
    test_bond_recon()
    #test_collated_bond_indexes()
    #test_bond_argmax()
