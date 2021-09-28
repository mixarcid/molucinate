from glob import glob
import sys
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def preprocess(folder):
    with open (f"{folder}/files.txt", "w") as out:
        for fname in glob(f"{folder}/zinc*.mol2"):
            mol = Chem.MolFromMol2File(fname)
            if mol is not None:
                out.write(f"{fname}\t{Chem.MolToSmiles(mol)}\n")

if __name__ == "__main__":
    preprocess(sys.argv[1])
