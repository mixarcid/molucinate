import torch
import hydra
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from torch.utils import data

try:
    from .tensor_mol import TensorMol, TMCfg
except ImportError:
    from tensor_mol import TensorMol, TMCfg

TT_SPLIT = 0.8
class PDBBindDataset(data.Dataset):

    def __init__(self, cfg, is_train, num_train=None):
        self.index = pd.read_table(cfg.platform.pdbbind_index, names=["pdb", "res", "year", "logK", "K", "NA", "ref", "name"], comment="#", sep="\s+")
        self.index = self.index.sample(frac=1, random_state=230).reset_index(drop=True)
        self.is_train = is_train
        if num_train is not None:
            self.num_train = num_train
            self.is_train = True
        else:
            self.num_train = int(len(self.index)*TT_SPLIT)

    def __len__(self):
        if self.is_train:
            return self.num_train
        else:
            return min(max(len(self.index) - self.num_train, 0), self.num_train)

    def __getitem__(self, index):
        if not self.is_train:
            index += self.num_train
        pname = self.index.pdb[index]
        mol_df = load_ligand_df(pname)
        molgrid, center = df2molgrid(mol_df)
        prot_df = load_pocket_df(pname)
        prot_mg, _ = df2molgrid(prot_df, center)

        lig_fname = get_ligand_path(pname)
        mol = Chem.MolFromMol2File(lig_fname)
        Chem.Kekulize(mol)

        graph_atoms = get_graph_atoms(mol, False)
        graph_edges = get_graph_edge_matrix(mol)
        
        return graph_atoms, graph_edges, molgrid, prot_mg


@hydra.main(config_path='../cfg', config_name='config.yaml')
def main(cfg):
    TMCfg.set_cfg(cfg.data)
    dataset = PDBBindDataset(cfg, False)
    print(dataset.index)
    print(len(dataset))
    for i, tmol in enumerate(dataset):
        render_kp_rt(tmol)
    
if __name__ == "__main__":
    from render import *
    main()
