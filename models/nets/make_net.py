from .basic import BasicEncoder, BasicDecoder
from .kp_rnn import KpRnnEncoder
from .atom_rnn import AtomRnnEncoder, AtomRnnDecoder
from .atom_kp_rnn import AtomKpRnnEncoder, AtomKpRnnDecoder

def make_encoder(hidden_size, cfg, gcfg):
    return {
        "kp_rnn": KpRnnEncoder,
        "basic": BasicEncoder,
        "atom_rnn": AtomRnnEncoder,
        "atom_kp_rnn": AtomKpRnnEncoder,
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
    
def make_decoder(hidden_size, cfg, gcfg):
    return {
        "kp_rnn": BasicDecoder,
        "basic": BasicDecoder,
        "atom_rnn": AtomRnnDecoder,
        "atom_kp_rnn": AtomKpRnnDecoder,
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
