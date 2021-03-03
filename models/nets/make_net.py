from .basic import BasicEncoder, BasicDecoder
from .kp_rnn import KpRnnEncoder

def make_encoder(hidden_size, cfg, gcfg):
    return {
        "kp_rnn": KpRnnEncoder,
        "basic": BasicEncoder
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
    
def make_decoder(hidden_size, cfg, gcfg):
    return {
        "kp_rnn": BasicDecoder,
        "basic": BasicDecoder
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
