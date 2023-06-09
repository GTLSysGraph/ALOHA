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



def build_easydict_nodecls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'S2GAE'
    MDT.MODEL.PARAM = {
        'log_steps'       : 1,
        'use_sage'    : 'GCN',
        'use_valedges_as_input':False,
        'num_layers':2,
        'decode_layers':2,
        'hidden_channels':128,
        'decode_channels':256,
        'dropout':0.2,
        'batch_size':8192,
        'lr':0.001,
        'epochs':400, 
        'seed':42,
        'eval_steps':1,
        'runs':1, 
        'mask_type' :'dm',  # help='dm | um')  # whether to use mask features    
        'patience'  : 50,       # help='Use attribute or not'                      
        'mask_ratio': 0.8

    }
    return MDT


# Attack-Citeseer dropout 0.2