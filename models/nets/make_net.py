from .basic import BasicEncoder, BasicDecoder
from .kp_rnn import KpRnnEncoder
from .atom_rnn import AtomRnnEncoder, AtomRnnDecoder
from .atom_kp_rnn import AtomKpRnnEncoder, AtomKpRnnDecoder
from .atn_net import AtnNetEncoder, AtnNetDecoder
from .ar_net import ArNetDecoder

def make_encoder(hidden_size, cfg, gcfg):
    return {
        "kp_rnn": KpRnnEncoder,
        "basic": BasicEncoder,
        "atom_rnn": AtomRnnEncoder,
        "atom_kp_rnn": AtomKpRnnEncoder,
        "atn_net": AtnNetEncoder,
        "ar_net": AtnNetEncoder
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
    
def make_decoder(hidden_size, cfg, gcfg):
    return {
        "kp_rnn": BasicDecoder,
        "basic": BasicDecoder,
        "atom_rnn": AtomRnnDecoder,
        "atom_kp_rnn": AtomKpRnnDecoder,
        "atn_net": AtnNetDecoder,
        "ar_net": ArNetDecoder
    }[cfg.net.name](hidden_size, cfg.net, gcfg)
