import torch
import hydra
from rdkit import Chem
from torch.utils import data

try:
    from .tensor_mol import TensorMol, TMCfg
except ImportError:
    from tensor_mol import TensorMol, TMCfg

TT_SPLIT = 0.9
class ZincDataset(data.Dataset):

    def __init__(self, cfg, is_train):
        self.files = open(cfg.platform.zinc_filtered_list).readlines()
        self.is_train = is_train
        self.num_train = int(len(self.files)*TT_SPLIT)
        self.zinc_dir = cfg.platform.zinc_dir
        if cfg.platform.stop_at is not None:
            self.num_train = min(cfg.platform.stop_at, self.num_train)
            self.is_train = True

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

        mol = Chem.MolFromMol2File(fname)
        return TensorMol(mol)

@hydra.main(config_path='../cfg', config_name='config.yaml')
def main(cfg):
    TMCfg.set_cfg(cfg.data)
    dataset = ZincDataset(cfg, False)
    print(len(dataset))
    for i, tmol in enumerate(dataset):
        render_molgrid_rt(tmol)
    
if __name__ == "__main__":
    from render import *
    main()
