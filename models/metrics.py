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

def get_recon_metrics(recon, x):
    ret = {}
    for fn in [rmsd]:
        ret[fn.__name__] = fn(recon, x)
    return ret

def get_gen_metrics(gen):
    rmsds = []
    topo_valids = []
    geom_valids = []
    for batch in range(gen.kps_1h.size(0)):
        mol = gen[batch].argmax().get_mol()
        topo_valid = 0.0
        geom_valid = 0.0
        if mol.GetNumAtoms() > 0:
            try:
                Chem.SanitizeMol(mol)
                topo_valid = 1.0
                try:
                    mol_uff = deepcopy(mol)
                    AllChem.UFFOptimizeMolecule(mol_uff, 500)
                    rmsd = Chem.rdMolAlign.AlignMol(mol, mol_uff)
                    if rmsd != 0:
                        rmsds.append(rmsd)
                        geom_valid = 1.0
                except RuntimeError:
                    pass
            except:
                pass
        rmsds.append(rmsd)
        geom_valids.append(geom_valid)
        topo_valids.append(topo_valid)
    return {
        "geom_valid": torch.mean(torch.tensor(geom_valids)),
        "topo_valid": torch.mean(torch.tensor(topo_valids)),
        "uff_rmsd": torch.mean(torch.tensor(rmsds))
    }

