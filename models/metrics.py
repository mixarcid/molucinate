import torch
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import sys
sys.path.insert(0, '../..')
from data.tensor_mol import TensorMol, TMCfg

def rmsd(recon, x):
    rets = []
    #todo: only rmsd actual atoms
    for batch in range(x.kps_1h.size(0)):
        sd = (recon[batch].get_coords() - x[batch].get_coords())**2
        rets.append(torch.sqrt(torch.mean(sd)))
    return torch.mean(torch.tensor(rets))

def frac_valid(gen):
    rets = []
    for batch in range(gen.kps_1h.size(0)):
        mol = gen[batch].argmax().get_mol()
        valid = 0.0
        try:
            Chem.SanitizeMol(mol)
            valid = 1.0
        except:
            pass
        rets.append(valid)
    return torch.mean(torch.tensor(rets))

def embed_rmsd(gen):
    rets = []
    for batch in range(gen.kps_1h.size(0)):
        mol = gen[batch].argmax().get_mol()
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        if mol.GetNumAtoms() > 0:
            #mol = Chem.AddHs(mol)
            mol_uff = deepcopy(mol)
            # for i in range(mol.GetNumAtoms()):
            #     print(i, mol.GetConformer().GetAtomPosition(i).x,
            #           mol.GetConformer().GetAtomPosition(i).y,
            #           mol.GetConformer().GetAtomPosition(i).z)
            #print(AllChem.UFFOptimizeMolecule(mol_uff, 500))
            # for i in range(mol_uff.GetNumAtoms()):
            #     print(i, mol_uff.GetConformer().GetAtomPosition(i).x,
            #           mol_uff.GetConformer().GetAtomPosition(i).y,
            #           mol_uff.GetConformer().GetAtomPosition(i).z)
            rmsd = Chem.rdMolAlign.AlignMol(mol, mol_uff)
            if rmsd != 0:
                rets.append(rmsd)
                print(rmsd)
    return torch.mean(torch.tensor(rets))

def get_recon_metrics(recon, x):
    ret = {}
    for fn in [rmsd]:
        ret[fn.__name__] = fn(recon, x)
    return ret

def get_gen_metrics(gen):
    ret = {}
    for fn in [frac_valid, embed_rmsd]:
        ret[fn.__name__] = fn(gen)
    return ret

