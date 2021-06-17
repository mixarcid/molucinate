import torch
import hydra
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from torch.utils import data

try:
    from .tensor_mol import TensorMol, TMCfg
    from .utils import rand_rotation_matrix
except ImportError:
    from tensor_mol import TensorMol, TMCfg
    from utils import rand_rotation_matrix

    
# pre_mol = Chem.MolFromMol2File('/home/boris/Data/Zinc/zinc483323.mol2')

TT_SPLIT = 0.9
class ZincDataset(data.Dataset):

    def __init__(self, cfg, is_train):
        self.files = np.array(open(f"{cfg.platform.zinc_dir}/files_filtered_{cfg.data.max_atoms}_{cfg.data.grid_dim}.txt").readlines())
        self.is_train = is_train
        self.num_train = int(len(self.files)*TT_SPLIT)
        self.zinc_dir = cfg.platform.zinc_dir
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

        try:
            mol = Chem.MolFromMol2File(fname)
            mat = rand_rotation_matrix()
            rdMolTransforms.TransformConformer(mol.GetConformer(0), mat)
            tm = TensorMol(mol)
        except:
            #raise
            mol = Chem.MolFromMol2File(fname)
            #print("Couldn't fit molecule; undoing rotation")
            tm = TensorMol(mol)
            
        return tm

@hydra.main(config_path='../cfg', config_name='config.yaml')
def main(cfg):
    TMCfg.set_cfg(cfg.data)
    dataset = ZincDataset(cfg, False)
    print(len(dataset))
    for i, tmol in enumerate(dataset):
        #print(tmol.atom_str())
        render_kp_rt(tmol)
    
if __name__ == "__main__":
    from render import *
    main()
