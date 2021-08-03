import hydra
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from copy import deepcopy
from random import random, choice, shuffle

try:
    from .tensor_mol import TensorMol, TMCfg
    from .utils import rand_rotation_matrix
    from .chem import *
except ImportError:
    from tensor_mol import TensorMol, TMCfg
    from utils import rand_rotation_matrix
    from chem import *

class MolAugment:

    def __init__(self, cfg):
        self.kekulize = cfg.data.kekulize
        self.use_kps = cfg.data.use_kps
        self.pos_randomize_std = cfg.data.pos_randomize_std
        self.atom_randomize_prob = cfg.data.atom_randomize_prob
        self.randomize_smiles = cfg.data.randomize_smiles

    def run(self, mol_og):

        if self.kekulize:
            Chem.Kekulize(mol_og)

        if self.randomize_smiles:
            indexes = list(range(mol_og.GetNumAtoms()))
            shuffle(indexes)
            mol_og = Chem.RenumberAtoms(mol_og, indexes)
            Chem.MolToSmiles(mol_og, canonical=False)
            order = eval(mol_og.GetProp("_smilesAtomOutputOrder"))
            mol_og = Chem.RenumberAtoms(mol_og, list(order))
        #else:
        #    Chem.MolToSmiles(mol_og, canonical=True)
        #    order = eval(mol_og.GetProp("_smilesAtomOutputOrder"))
        #    mol_og = Chem.RenumberAtoms(mol_og, list(order))
            
            
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
    cfg.data.max_atoms = 64
    cfg.data.grid_dim = 32
    TMCfg.set_cfg(cfg.data)
    aug = MolAugment(cfg)
    mol = Chem.MolFromMol2File('test_data/zinc100001.mol2')
    print(Chem.MolToSmiles(mol, canonical=False))
    tm, tm_r = aug.run(mol)
    mol2 = tm.get_mol()
    print(Chem.MolToSmiles(mol2, canonical=False))

if __name__ == "__main__":
    main()
