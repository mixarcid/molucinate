from .basic import BasicEncoder, BasicDecoder
from .kp_rnn import KpRnnEncoder
from .atom_rnn import AtomRnnEncoder, AtomRnnDecoder

def make_encoder(hidden_size, cfg, gcfg):
    return {
        "kp_rnn": KpRnnEncoder,
        "basic": BasicEncoder,
        "atom_rnn": AtomRnnEncoder
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
    
def make_decoder(hidden_size, cfg, gcfg):
    return {
        "kp_rnn": BasicDecoder,
        "basic": BasicDecoder,
        "atom_rnn": AtomRnnDecoder
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
