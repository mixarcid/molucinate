import torch
import hydra
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from torch.utils import data
from copy import deepcopy
from random import random, choice
import numpy as np

try:
    from .tensor_mol import TensorMol, TMCfg
    from .utils import rand_rotation_matrix
    from .chem import *
except ImportError:
    from tensor_mol import TensorMol, TMCfg
    from utils import rand_rotation_matrix
    from chem import *
    
# pre_mol = Chem.MolFromMol2File('/home/boris/Data/Zinc/zinc483323.mol2')

TT_SPLIT = 0.9
class ZincDataset(data.Dataset):

    def __init__(self, cfg, is_train):
        self.files = np.array(open(f"{cfg.platform.zinc_dir}/files_filtered_{cfg.data.max_atoms}_{cfg.data.grid_dim}.txt").readlines())
        self.is_train = is_train
        self.num_train = int(len(self.files)*TT_SPLIT)
        self.zinc_dir = cfg.platform.zinc_dir
        self.kekulize = cfg.data.kekulize
        self.use_kps = cfg.data.use_kps
        self.pos_randomize_std = cfg.data.pos_randomize_std
        self.atom_randomize_prob = cfg.data.atom_randomize_prob
        if cfg.debug.stop_at is not None:
            self.num_train = min(cfg.debug.stop_at, self.num_train)
            #self.is_train = True

    def __len__(self):
        if self.is_train:
            return self.num_train
        else:
            return min(max(len(self.files) - self.num_train, 0), self.num_train)

    def __getitem__(self, index):
        if not self.is_train:
            index += self.num_train
        fname, smiles = self.files[index].strip().split('\t')
        fname = self.zinc_dir + fname.split("/")[-1]

        mol_og = Chem.MolFromMol2File(fname)
        if self.kekulize:
            Chem.Kekulize(mol_og)
        mol = deepcopy(mol_og)

        if self.use_kps:
            try:
                mat = rand_rotation_matrix()
                rdMolTransforms.TransformConformer(mol.GetConformer(0), mat)
                tm = TensorMol(mol)
            except:
                #raise
                tm = TensorMol(mol_og)
                #print("Couldn't fit molecule; undoing rotation")

            try:
                mol_random = self.randomize_pos(mol)
                tm_random = TensorMol(mol_random)
            except:
                tm_random = deepcopy(tm)
        else:
            tm = TensorMol(mol_og)
            tm_random = deepcopy(tm)

        for i in range(tm_random.atom_types.size(0)):
            if random() < self.atom_randomize_prob:
                tm_random.atom_types[i] = choice(list(ATOM_TYPE_HASH.values()))
            if random() < self.atom_randomize_prob:
                tm_random.bonds.atom_valences[i] = choice(list(range(TMCfg.max_valence+1)))
            
        return tm, tm_random

    def randomize_pos(self, mol):
        mol_random = deepcopy(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            pos = mol_random.GetConformer().GetAtomPosition(i)
            pos.x += np.random.normal(0.0, self.pos_randomize_std)
            pos.y += np.random.normal(0.0, self.pos_randomize_std)
            pos.z += np.random.normal(0.0, self.pos_randomize_std)
            mol_random.GetConformer().SetAtomPosition(i, pos)
        return mol_random

@hydra.main(config_path='../cfg', config_name='config.yaml')
def main(cfg):
    TMCfg.set_cfg(cfg.data)
    dataset = ZincDataset(cfg, False)
    print(len(dataset))
    for i, (tmol, tmol_r) in enumerate(dataset):
        print(tmol.atom_str())
        if cfg.data.use_kps:
            render_kp_rt(tmol_r)
        """img = render_tmol(tmol, dims=(600, 600))
        cv2.imwrite(f"test_output/zinc_{i}.png", img)
        for j in range(TMCfg.max_atoms):
            tm2 = deepcopy(tmol)
            tm2.atom_types[:j] = ATOM_TYPE_HASH['_']
            tm2.atom_types[j+1:] = ATOM_TYPE_HASH['_']
            img = render_kp(tm2, dims=(600, 600))
            cv2.imwrite(f"test_output/za_{i}_{j}.png", img)"""
        if i > 10:
            break
    
if __name__ == "__main__":
    from render import *
    import cv2
    from copy import deepcopy
    main()
