from .basic import BasicEncoder, BasicDecoder

def make_encoder(cfg, gfcg):
    return {
        "basic": BasicEncoder
    }[cfg.encoder.name](cfg.encoder, gcfg)
    
def make_decoder(cfg, gcfg):
    return {
        "basic": BasicDecoder
    }[cfg.decoder.name](cfg.decoder, gcfg)
