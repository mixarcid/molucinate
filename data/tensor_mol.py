from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem.MolStandardize import rdMolStandardize

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

INIT_DEVICE = 'cpu'

class TMCfg:

    max_atoms = None
    grid_dim = None
    grid_coords = None
    grid_size = None
    grid_shape = None
    max_valence = None
    use_kps = None

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
        TMCfg.use_kps = cfg.use_kps
    

class TensorBonds(Collatable):
    bond_types: torch.Tensor
    bonded_atoms: torch.Tensor
    atom_valences: torch.Tensor

    def __init__(self, mol=None,
                 bond_types=None,
                 bonded_atoms=None,
                 atom_valences=None,
                 device='cpu'):
        
        self.cached_indexes = None
        
        if mol is None:
            self.bond_types = bond_types
            self.bonded_atoms = bonded_atoms
            self.atom_valences = atom_valences
            return

        self.bond_types = torch.zeros((TMCfg.max_atoms, TMCfg.max_valence), dtype=torch.long, device=INIT_DEVICE) #NUM_BOND_TYPES
        self.bonded_atoms = torch.zeros((TMCfg.max_atoms, TMCfg.max_valence), dtype=torch.long, device=INIT_DEVICE) #TMCfg.max_atoms
        self.atom_valences = torch.zeros((TMCfg.max_atoms, NUM_ACT_BOND_TYPES), dtype=torch.long, device=INIT_DEVICE)

        tot_valences = torch.zeros((TMCfg.max_atoms), dtype=torch.long, device=INIT_DEVICE)
        
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            if start_idx > end_idx:
                tmp = end_idx
                end_idx = start_idx
                start_idx = tmp
            cur_valence = tot_valences[end_idx]
            btype = BOND_TYPE_HASH[bond.GetBondType()]
            self.bond_types[end_idx][cur_valence] = btype
            self.bonded_atoms[end_idx][cur_valence] = start_idx
            tot_valences[end_idx] += 1
            self.atom_valences[end_idx][btype-1] += 1
            self.atom_valences[start_idx][btype-1] += 1

    def argmax(self, atom_types):
        assert self.bond_types.dtype == torch.float and self.bonded_atoms.dtype == torch.float
        bond_types = torch.argmax(self.bond_types, -1)
        bonded_atoms = torch.argmax(self.bonded_atoms, -1)
        atom_valences = torch.argmax(self.atom_valences, -1)
        return TensorBonds(None, bond_types, bonded_atoms, atom_valences)

    def get_all_indexes(self):
        if self.cached_indexes is not None:
            return self.cached_indexes
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
        self.cached_indexes = out
        return out

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
        self.coords = torch.zeros((TMCfg.max_atoms, 3), device=INIT_DEVICE)
        # batch, atom_idx, atom_type
        self.atom_types = torch.zeros(TMCfg.max_atoms, dtype=torch.long, device=INIT_DEVICE)
        # batch, atom_idx, atom_valence
        #self.atom_valences = torch.zeros((TMCfg.max_atoms, NUM_ACT_BOND_TYPES), dtype=torch.long, , device=INIT_DEVICE)

        for i, atom in enumerate(mol.GetAtoms()):
            charge = atom.GetFormalCharge()
            # assert charge == 0
            atom_type = ATOM_TYPE_HASH[atom.GetSymbol()]
            self.atom_types[i] = atom_type
            pos = mol.GetConformer().GetAtomPosition(i)
            self.coords[i] = torch.tensor([ pos.x, pos.y, pos.z ])
        self.center_coords()

        self.bonds = TensorBonds(mol)
            

    def center_coords(self):
        min_coord = torch.zeros((3,), device=INIT_DEVICE)
        max_coord = torch.zeros((3,), device=INIT_DEVICE)
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
            self.bonds = bonds
            return
        tmb = TensorMolBasic(mol)
        self.atom_types = tmb.atom_types
        self.bonds = tmb.bonds

        if TMCfg.use_kps:
            sz = TMCfg.grid_size
            # batch, atom_idx, width, height, depth
            kp_shape = (TMCfg.max_atoms, sz, sz, sz)
            self.kps = torch.zeros(kp_shape, device=INIT_DEVICE)
            self.kps_1h = torch.zeros(kp_shape, device=INIT_DEVICE)
            # batch, atom_type, width, height, depth
            self.molgrid = torch.zeros((NUM_ATOM_TYPES, sz, sz, sz), device=INIT_DEVICE)
            for i, (coord, atom) in enumerate(zip(tmb.coords, self.atom_types)):
                if atom == ATOM_TYPE_HASH['_']: break
                grid = self.gridify_atom(coord, atom)
                self.kps[i] = grid
                self.kps_1h[i] = self.gridify_atom(coord, atom, True)
                self.molgrid[atom] += grid
        else:
            self.kps = None
            self.kps_1h = None
            self.molgrid = None

    def gridify_atom(self, coord, atom, should_1h=False):
        x, y, z = coord*2 + TMCfg.grid_dim
        center_index = (int(y), int(x), int(z))
        if should_1h:
            A = torch.zeros(TMCfg.grid_shape)
            A[center_index] = 1
        else:
            r = ATOM_RADII_LIST[atom]
            indexes = []
            vals = []
            for xdiff in range(-int(r*3) - 1, int(r*3) + 1):
                x = center_index[0] + xdiff
                if x >= TMCfg.grid_dim*2 or x < 0: continue
                for ydiff in range(-int(r*3) - 1, int(r*3) + 1):
                    y = center_index[1] + ydiff
                    if y >= TMCfg.grid_dim*2 or y < 0: continue
                    for zdiff in range(-int(r*3) - 1, int(r*3) + 1):
                        z = center_index[2] + zdiff
                        if z >= TMCfg.grid_dim*2 or z < 0: continue
                        index = (x, y, z)
                        diff = TMCfg.grid_coords[index] - coord.cpu().numpy()
                        d2 = np.dot(diff, diff)
                        if d2 > (1.5*r)**2: continue
                        
                        dist = np.sqrt(d2)
                        if dist < r:
                            val = np.exp(-2*(d2/(r**2)))
                        else:
                            val = (4/((np.e**2)*(r**2)))*d2 - (12/((np.e**2)*r))*dist + 9/(np.e**2)
                        indexes.append(index)
                        vals.append(val)
            
            sz = TMCfg.grid_size
            #print(len(indexes), indexes[0])
            #print(len(vals), vals[0])
            indexes = torch.tensor(indexes, dtype=torch.long).T
            A = torch.sparse_coo_tensor(indexes, vals, size=(sz, sz, sz)).to_dense()
        return A

    def argmax(self):
        if self.atom_types.dtype != torch.long:
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
        if self.kps_1h is None: return mol
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

def empty_mol():
    sz = TMCfg.grid_size
    kp_shape = (TMCfg.max_atoms, sz, sz, sz)
    kps = torch.zeros(kp_shape, device=INIT_DEVICE)
    kps_1h = torch.zeros(kp_shape, device=INIT_DEVICE)
    molgrid = torch.zeros((NUM_ATOM_TYPES, sz, sz, sz), device=INIT_DEVICE)
    atom_types = torch.zeros(TMCfg.max_atoms, dtype=torch.long, device=INIT_DEVICE)
    bond_types = torch.zeros((TMCfg.max_atoms, TMCfg.max_valence), dtype=torch.long, device=INIT_DEVICE) #NUM_BOND_TYPES
    bonded_atoms = torch.zeros((TMCfg.max_atoms, TMCfg.max_valence), dtype=torch.long, device=INIT_DEVICE) #TMCfg.max_atoms
    atom_valences = torch.zeros((TMCfg.max_atoms, NUM_ACT_BOND_TYPES), dtype=torch.long, device=INIT_DEVICE)
    return TensorMol(kps=kps,
                     kps_1h=kps_1h,
                     molgrid=molgrid,
                     atom_types=atom_types,
                     bonds=TensorBonds(bond_types=bond_types,
                                       bonded_atoms=bonded_atoms,
                                       atom_valences=atom_valences))
                     
    
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
    #print(torch.amax(tm.molgrid))
    print(tm.bonds.atom_valences)

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
    for n in range(3):
        #order = Chem.CanonicalRankAtoms(mol, includeChirality=True)
        Chem.MolToSmiles(mol, canonical=True)
        
        print(Chem.MolToSmiles(mol, canonical=True))
        order = eval(mol.GetProp("_smilesAtomOutputOrder"))
        print(order)
        #order = mol.GetPropsAsDict(includePrivate=True, 
        #                           includeComputed=True)['_smilesAtomOutputOrder']
        mol = Chem.RenumberAtoms(mol, list(order))
        print(TensorMol(mol).atom_str())
        #print(Chem.MolToSmiles(mol))
        print("")
    
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
    #test_basic()
    #test_mol_export()
    test_bond_recon()
    #test_collated_bond_indexes()
    #test_bond_argmax()
