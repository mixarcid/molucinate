import numpy as np
from rdkit import Chem

ATOM_TYPE_LIST = [ '_', '^', 'C', 'O', 'N', 'H', 'P', 'Cl', 'F', 'Br', 'S', 'I' ]
ATOM_TYPE_HASH = { atom: i for i, atom in enumerate(ATOM_TYPE_LIST) }
NUM_ATOM_TYPES = len(ATOM_TYPE_LIST)

BOND_TYPE_LIST = [ '_', Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC ]
BOND_TYPE_HASH = { bond: i for i, bond in enumerate(BOND_TYPE_LIST) }
NUM_BOND_TYPES = len(BOND_TYPE_LIST)
NUM_ACT_BOND_TYPES = NUM_BOND_TYPES - 1

ATOM_RADII_HASH = {
    'H': 1.1,
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'P': 1.8,
    'S': 1.8,
    'Cl': 1.75,
    'F': 1.47,
    'Br': 1.85,
    'I': 1.98
}
ATOM_RADII_LIST = np.zeros((NUM_ATOM_TYPES,))
for atom, r in ATOM_RADII_HASH.items():
    ATOM_RADII_LIST[ATOM_TYPE_HASH[atom]] = r

ATOM_COLORS = {
    '^': np.array([127, 127, 127]),
    '_': np.array([127, 127, 127]),
    'C': np.array([0,0,0]),
    'H': np.array([255,255,255]),
    'N': np.array([0,0,255]),
    'O': np.array([255,0,0]),
    'P': np.array([203,75,22]),
    'S': np.array([255,255,0]),
    'Cl': np.array([0,255,0]),
    'F': np.array([0,0,127]),
    'Br': np.array([210,105,30]),
    'I': np.array([255, 0, 255])
}
