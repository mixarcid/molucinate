import torch
import hydra
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from torch.utils import data
from copy import deepcopy
import traceback

try:
    from .tensor_mol import TensorMol, TMCfg
    from .utils import rand_rotation_matrix
    from .chem import *
except ImportError:
    from tensor_mol import TensorMol, TMCfg
    from utils import rand_rotation_matrix
    from chem import *

TT_SPLIT = 0.8
class PDBBindDataset(data.Dataset):

    def __init__(self, cfg, is_train):
        self.refined_dir = cfg.platform.pdbbind_refined_dir
        self.index = pd.read_table(cfg.platform.pdbbind_index, names=["pdb", "res", "year", "logK", "K", "NA", "ref", "name"], comment="#", sep="\s+")
        self.index = self.index.sample(frac=1, random_state=230).reset_index(drop=True)
        self.kekulize = cfg.data.kekulize
        self.canonicalize = cfg.data.canonicalize
        self.use_kps = cfg.data.use_kps
        self.is_train = is_train
        self.num_train = int(len(self.index)*TT_SPLIT)
        if cfg.debug.stop_at is not None:
            self.num_train = min(cfg.debug.stop_at, self.num_train)

    def __len__(self):
        if self.is_train:
            return self.num_train
        else:
            return min(max(len(self.index) - self.num_train, 0), self.num_train)

    def __getitem__(self, index):
        if not self.is_train:
            index += self.num_train
        pname = self.index.pdb[index]
        ligand_fname = f"{self.refined_dir}{pname}/{pname}_ligand.sdf"
        ligand_og = next(Chem.SDMolSupplier(ligand_fname))
        #ligand_og = Chem.MolFromMol2File(ligand_fname)
        ligand = deepcopy(ligand_og)
        
        if self.kekulize:
            Chem.Kekulize(ligand_og)
        if self.canonicalize:
            order = Chem.CanonicalRankAtoms(ligand_og, includeChirality=True)
            ligand_og = Chem.RenumberAtoms(ligand_og, list(order))

            ligand = deepcopy(ligand_og)

        if self.use_kps:
            try:
                mat = rand_rotation_matrix()
                rdMolTransforms.TransformConformer(ligand.GetConformer(0), mat)
                tm_lig = TensorMol(ligand)
            except:
                tm_lig = TensorMol(ligand_og)
        else:
            tm_lig = TensorMol(ligand_og)

        return tm_lig


@hydra.main(config_path='../cfg', config_name='config.yaml')
def main(cfg):
    TMCfg.set_cfg(cfg.data)
    dataset = PDBBindDataset(cfg, False)
    print(dataset.index)
    print(len(dataset))
    for i in range(len(dataset)):
        try:
            tmol = dataset[i]
            print("success")
            render_kp_rt(tmol)
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
    
if __name__ == "__main__":
    from render import *
    main()
