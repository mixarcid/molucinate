from .vae import VAE

def make_model(cfg):
    return {
        "vae": VAE
    }[cfg.model.name](cfg.model, cfg)
