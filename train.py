import hydra
import pytorch_lightning as pl

@hydra.main(config_path='cfg', config_name="config")
def train(cfg):
    print(cfg)
    
if __name__ == '__main__':
    train()
