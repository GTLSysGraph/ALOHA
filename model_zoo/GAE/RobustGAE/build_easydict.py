
from easydict import EasyDict

def build_easydict_nodecls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'RobustGAE'
    MDT.MODEL.PARAM = {
        'seeds'         : [213,89,1234,74],
        "device"        : -1 ,
        "max_epoch"     : 200,             
        "warmup_steps"  : -1,
        "num_heads"     : 4 ,
        "num_out_heads" : 1,            
        "num_layers"    : 2,
        "num_hidden"    : 256,           
        "residual"      : False,
        "in_drop"       : 0.2,              
        "attn_drop"     : 0.1,   
        "norm"          : None ,
        "lr"            : 0.005,
        "weight_decay"  : 5e-4,                   
        "negative_slope": 0.2,                        
        "activation"    : "prelu",
        "mask_rate"     : 0.5,
        "drop_edge_rate": 0.0,
        "replace_rate"  : 0.0,
        "encoder"       : "gat",
        "decoder"       : "gat",
        "loss_fn"       : "sce",
        "alpha_l"       : 2,
        "optimizer"     : "adam", 
        "max_epoch_f"   : 30,
        "lr_f"          : 0.001,
        "weight_decay_f": 0.0,
        "linear_prob"   : False,
    
        "load_model"    : False,
        "save_model"    : False,
        "use_cfg"       : True,
        "logging"       : False,
        "scheduler"     : False,
        "concat_hidden" : False,

        # for graph classification
        "pooling"    : "mean",
        "deg4feat"   : False,
        "batch_size" :32
    }

    return MDT