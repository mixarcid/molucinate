import torch
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from sklearn.metrics import jaccard_score

import sys
sys.path.insert(0, '../..')
from data.chem import *
from data.tensor_mol import TensorMol, TMCfg

def rmsd(recon, x):
    rets = []
    idxs = x.atom_types != ATOM_TYPE_HASH["_"]
    #todo: only rmsd actual atoms
    for batch in range(x.kps_1h.size(0)):
        idx = idxs[batch]
        sd = (recon[batch].get_coords()[idx] - x[batch].get_coords()[idx])**2
        rets.append(torch.sqrt(torch.mean(sd)))
    return torch.mean(torch.tensor(rets))

def bond_iou(recon, x):
    rets = []
    for batch in range(x.bonds.data.size(0)):
        rets.append(jaccard_score(
            recon[batch].argmax().bonds.data.reshape((NUM_BOND_TYPES, -1)).cpu().numpy().T,
            x[batch].bonds.data.reshape((NUM_BOND_TYPES, -1)).cpu().numpy().T,
            average='macro',
            zero_division=1
        ))
    return torch.mean(torch.tensor(rets))

def perfect_topo_acc(recon, x):
    rets = []
    for batch in range(x.atom_types.size(0)):
        rb = recon[batch].argmax()
        xb = x[batch]
        idxs = xb.atom_types != ATOM_TYPE_HASH["_"]
        indexes_same = rb.bonds.get_all_indexes() == xb.bonds.get_all_indexes()
        rets.append(float(indexes_same and (rb.atom_types == xb.atom_types).all()))
    return torch.mean(torch.tensor(rets))

def get_recon_metrics(recon, x):
    ret = {}
    fns = [ perfect_topo_acc ]
    if TMCfg.use_kps:
        fns += [ rmsd ]
    for fn in fns:
        ret[fn.__name__] = fn(recon, x)
    return ret

def get_gen_metrics(gen):
    if gen.kps_1h is None:
        return {}
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
        geom_valids.append(geom_valid)
        topo_valids.append(topo_valid)
    return {
        "geom_valid": torch.mean(torch.tensor(geom_valids)),
        "topo_valid": torch.mean(torch.tensor(topo_valids)),
        "uff_rmsd": torch.mean(torch.tensor(rmsds))
    }

