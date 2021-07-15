import pandas as pd
import hydra
from shutil import copyfile

def create_initial_csv(cfg):
    index = pd.read_table(cfg.platform.crossdocked_index, names=["label", "affinity", "rmsd", "rec_file", "lig_file"], comment="#", sep="\s+")
    idxs = index['lig_file'].str.endswith("_docked_0.gninatypes")
    index = index[idxs]
    print(len(index))
    index.to_csv('crossdocked.csv')

def filter_index(cfg):
    index = pd.read_csv('crossdocked.csv')
    folder_names = index['lig_file'].str.split('/').str[0]
    lig_names = index['lig_file'].str.split('_rec_').str[-1].str.split('_lig').str[0]
    index['lig_file'] = folder_names + '/' + lig_names
    index['rec_file'] = index['rec_file'].str[:-13] + '.pdb'
    index.to_csv('crossdocked_filtered.csv')

@hydra.main(config_path='cfg', config_name="config")
def gen_crossdocked(cfg):
    index = pd.read_csv('crossdocked_filtered.csv')
    for i, line in index.iterrows():
        if line['affinity'] == 0: continue
        rec_fname = cfg.platform.crossdocked_dir + line['rec_file']
        new_rec_fname = cfg.platform.crossdocked_filtered_dir + '-'.join(line['rec_file'].split('/'))
        lig_fnames = [ cfg.platform.crossdocked_dir + line['lig_file'] + ext for ext in ['_uff2.sdf', '_uff.sdf', '_lig.pdb']]
        new_lig_fnames = [ cfg.platform.crossdocked_filtered_dir + '-'.join(line['lig_file'].split('/')) + ext for ext in ['_uff2.sdf', '_uff.sdf', '_lig.pdb']]
        try:
            copyfile(rec_fname, new_rec_fname)
        except:
            print(f"Error copying {rec_fname} to {new_rec_fname}")

        for lig_fname, new_lig_fname in zip(lig_fnames, new_lig_fnames):
            try:
                copyfile(lig_fname, new_lig_fname)
                break
            except:
                print(f"Error copying {lig_fname} to {new_lig_fname}")
                

if __name__ == "__main__":
    gen_crossdocked()
