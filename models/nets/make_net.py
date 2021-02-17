from .basic import BasicEncoder, BasicDecoder

def make_encoder(hidden_size, cfg, gcfg):
    return {
        "basic": BasicEncoder
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
    
def make_decoder(hidden_size, cfg, gcfg):
    return {
        "basic": BasicDecoder
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
