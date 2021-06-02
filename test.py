import neptune.new as neptune
import hydra

@hydra.main(config_path='cfg', config_name="config")
def test(cfg):

    run = neptune.init(project="mixarcid/molucinate",
                       run=cfg.test_run_id)
    path = f"{cfg.platform.results_path}weights.pt"
    run["artifacts/weights.pt"].download(path)

if __name__ == "__main__":
    test()
