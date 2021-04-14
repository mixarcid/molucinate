from tqdm import tqdm
from rdkit import Chem
import hydra

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from data.tensor_mol import TensorMol, TensorMolBasic, TMCfg

@hydra.main(config_path='cfg', config_name="config")
def main(cfg):
    print_except = False
    num_files = 0
    TMCfg.set_cfg(cfg.data)
    with open(f"{cfg.platform.zinc_dir}/files_filtered_{cfg.data.max_atoms}_{cfg.data.grid_dim}.txt", "w") as f:
        with open(f"{cfg.platform.zinc_dir}/files.txt", "r") as zinc_list:
            for line in tqdm(zinc_list.readlines()):
                fname, smiles = line.strip().split("\t")
                try:
                    mol = Chem.MolFromMol2File(cfg.platform.zinc_dir + fname)
                    tm = TensorMolBasic(mol)
                    f.write(line)
                    num_files += 1
                except Exception as e:
                    if print_except:
                        print(smiles, repr(e))
                except KeyboardInterrupt:
                    raise
    print(f"Processed {num_files} files.")

if __name__ == "__main__":
    main()
