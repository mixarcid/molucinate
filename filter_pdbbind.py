from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import pandas as pd
import hydra
import traceback

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from data.tensor_mol import TensorMol, TensorMolBasic, TMCfg

@hydra.main(config_path='cfg', config_name="config")
def main(cfg):
    print_except = False
    num_files = 0
    TMCfg.set_cfg(cfg.data)
    
    index = pd.read_table(cfg.platform.pdbbind_index, names=["pdb", "res", "year", "logK", "K", "NA", "ref", "name"], comment="#", sep="\s+")

    num_files = 0
    for i in tqdm(range(len(index))):
        try:
            pname = index.pdb[i]
            ligand_fname = "/home/boris/Downloads/1km3_up6_uff2.sdf" #f"{cfg.platform.pdbbind_refined_dir}{pname}/{pname}_ligand.sdf"
            ligand_og = next(Chem.SDMolSupplier(ligand_fname))
            Chem.Kekulize(ligand_og)

            smiles = Chem.MolToSmiles(ligand_og)
            if 'P' in smiles:
                print(smiles, ligand_fname)
            
            tm = TensorMolBasic(ligand_og)
            num_files += 1
        except Exception:
            if print_except:
                print(f"Error processing {ligand_fname}:")
                traceback.print_exc()
        except KeyboardInterrupt:
            raise
        
    print(f"Processed {num_files} files out of {len(index)}")

if __name__ == "__main__":
    main()
