import pandas as pd
import hydra

@hydra.main(config_path='cfg', config_name="config")
def gen_crossdocked(cfg):
    index = pd.read_table(cfg.platform.crossdocked_index, names=["label", "affinity", "rmsd", "rec_file", "lig_file"], comment="#", sep="\s+")
    print(index.head)

if __name__ == "__main__":
    gen_crossdocked()
