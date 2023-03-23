from easydict import EasyDict
import yaml
import logging

def load_best_configs(param, dataset_name, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if dataset_name not in configs:
        logging.info("Best args not found")
        return param

    logging.info("Using best configs")
    configs = configs[dataset_name]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(param, k, v)
    print("------ Use best configs ------")
    return param



def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'GRACE'
    MDT.MODEL.PARAM = {
        "use_cfg"       : True,
    }
    return MDT