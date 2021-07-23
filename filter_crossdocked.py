from tqdm import tqdm
from rdkit import Chem
import hydra
import pandas as pd

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from data.tensor_mol import TensorMol, TensorMolBasic, TMCfg
from data.render import *

@hydra.main(config_path='cfg', config_name="config")
def main(cfg):
    print_except = False
    num_files = 0
    TMCfg.set_cfg(cfg.data)
    index = pd.read_csv(cfg.platform.crossdocked_crystal_index)
    seen = {}
    with open(f"{cfg.platform.crossdocked_filtered_dir}/files_filtered_{cfg.data.max_atoms}_{cfg.data.grid_dim}.txt", "w") as f:
        for i, line in tqdm(index.iterrows(), total=len(index)):
            rec_fname = line["rec_file"]
            lig_fname = line["lig_file"]
            pdb_rec = rec_fname.split("_")[-3]
            pdb_lig = lig_fname.split("_")[-3]
            affinity = abs(line["affinity"])
            if pdb_rec != pdb_lig: continue
            if (rec_fname, lig_fname) in seen:
                assert seen[(rec_fname, lig_fname)] == affinity
                continue
            seen[(rec_fname, lig_fname)] = affinity
                
            try:
                
                ligand_og = next(Chem.SDMolSupplier(lig_fname))
                Chem.Kekulize(ligand_og)

                tm = TensorMolBasic(ligand_og)

                #tm = TensorMol(ligand_og)
                #render_kp_rt(tm)

                num_files += 1
                
            except Exception as e:
                if print_except:
                    print(repr(e))
            except KeyboardInterrupt:
                raise
    print(f"Processed {num_files} files.")

if __name__ == "__main__":
    main()
