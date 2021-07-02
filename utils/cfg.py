from collections import defaultdict
from omegaconf import OmegaConf, DictConfig
import neptune.new as neptune

def count_keys(params):
    ret = defaultdict(int)
    for key, value in params.items():
        if isinstance(value, dict):
            for key2, count in count_keys(value).items():
                ret[key2] += count
        else:
            ret[key] += 1
    return ret

def flatten_dict(params, key_counts=None):
    if key_counts is None:
        key_counts = count_keys(params)
    ret = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for key2, val2 in flatten_dict(value).items():
                if key_counts[key2] == 1 and not '.' in key2:
                    ret[key2] = val2
                else:
                    ret[f"{key}.{key2}"] = val2
        else:
            ret[key] = value
    return ret

def get_params(cfg):
    params = OmegaConf.to_container(cfg, True)
    for key in ["debug", "platform"]:
        del params[key]
    params = flatten_dict(params)
    return params

def copy_template_cfg(cfg, new_cfg):
    for key in cfg:
        if key in new_cfg:
            if isinstance(cfg[key], DictConfig):
                copy_template_cfg(cfg[key], new_cfg[key])
            else:
                cfg[key] = new_cfg[key]

def get_checkpoint_cfg(cfg, run_id, use_cache=False):
    ckpt_path = f"{cfg.platform.results_path}{run_id}_weights.ckpt"
    cfg_path = f"{cfg.platform.results_path}{run_id}_cfg.yaml"
    
    if not use_cache:
        print(f"Downloading latest {run_id} checkpoint")
        run = neptune.init(project="mixarcid/molucinate",
                           run=run_id)
    
        run["artifacts/weights.ckpt"].download(ckpt_path)
        run["artifacts/cfg.yaml"].download(cfg_path)
        
    new_cfg = OmegaConf.load(cfg_path)
    new_cfg.platform = cfg.platform
    copy_template_cfg(cfg, new_cfg)
    
    return cfg, ckpt_path
