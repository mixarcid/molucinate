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
    from .tensor_mol import TensorMol, TMCfg, empty_mol
    from .mol_augment import MolAugment
    from .utils import rand_rotation_matrix
    from .chem import *
except ImportError:
    from tensor_mol import TensorMol, TMCfg, empty_mol
    from utils import rand_rotation_matrix
    from mol_augment import MolAugment
    from chem import *
    
# pre_mol = Chem.MolFromMol2File('/home/boris/Data/Zinc/zinc483323.mol2')

TT_SPLIT = 0.9
class ZincDataset(data.Dataset):

    def __init__(self, cfg, is_train):
        self.is_train = is_train
        self.profile = cfg.debug.profile
        self.files = np.array(open(f"{cfg.platform.zinc_dir}/files_filtered_{cfg.data.max_atoms}_{cfg.data.grid_dim}.txt").readlines())
        self.num_train = int(len(self.files)*TT_SPLIT)
        self.zinc_dir = cfg.platform.zinc_dir
        self.augment = MolAugment(cfg)
        if cfg.debug.stop_at is not None:
            self.num_train = min(cfg.debug.stop_at, self.num_train)
            #self.is_train = True

    def __len__(self):
        if self.is_train:
            return self.num_train
        else:
            return min(max(len(self.files) - self.num_train, 0), self.num_train)

    def get_smiles_set(self):
        ret = set()
        for i in range(len(self)):
            fname, smiles = self.files[i].strip().split('\t')
            ret.add(smiles)
        return ret
    
    def __getitem__(self, index):
        if not self.is_train:
            index += self.num_train
        fname, smiles = self.files[index].strip().split('\t')
        fname = self.zinc_dir + fname.split("/")[-1]

        mol_og = Chem.MolFromMol2File(fname)
        return self.augment.run(mol_og)
        
@hydra.main(config_path='../cfg', config_name='config.yaml')
def main(cfg):
    TMCfg.set_cfg(cfg.data)
    dataset = ZincDataset(cfg, False)
    print(len(dataset))
    for i, (tmol, tmol_r) in enumerate(dataset):
        print(tmol.atom_str())
        if cfg.data.use_kps:
            render_molgrid_rt(tmol)
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
