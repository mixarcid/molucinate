from tqdm import tqdm
from rdkit import Chem
from glob import glob
import hydra

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from data.tensor_mol import TensorMol, TensorMolBasic, TMCfg

@hydra.main(config_path='cfg', config_name="config")
def main(cfg):
    print_except = False
    num_files = 0
    TMCfg.set_cfg(cfg.data)
    files = glob(f"{cfg.platform.qm9_dir}/*.mol2")
    with open(f"{cfg.platform.qm9_dir}/files_filtered_{cfg.data.max_atoms}_{cfg.data.grid_dim}.txt", "w") as f:
        for fname in tqdm(files):
            try:
                mol = Chem.MolFromMol2File(fname)
                tm = TensorMolBasic(mol)
                smiles = Chem.MolToSmiles(mol)
                f.write(f"{fname}\t{smiles}\n")
                num_files += 1
            except Exception as e:
                if print_except:
                    print(smiles, repr(e))
            except KeyboardInterrupt:
                raise
    print(f"Processed {num_files} files.")

if __name__ == "__main__":
    main()
