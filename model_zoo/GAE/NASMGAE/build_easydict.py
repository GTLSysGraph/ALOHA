
from easydict import EasyDict

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'GraphMAE'
    MDT.MODEL.PARAM = {
        'seeds'         : [12,34,6,7], # 0.8367±0.0050
        "device"        : -1 ,
        "max_epoch"     : 200,             
        "warmup_steps"  : -1,
        "num_heads"     : 2 ,
        "num_out_heads" : 1,            
        "num_layers"    : 2,
        "num_hidden"    : 256,           
        "residual"      : False,
        "in_drop"       : 0.2,              
        "attn_drop"     : 0.1,   
        "norm"          : None ,
        "lr"            : 0.00015,
        "weight_decay"  : 0.0,                   
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
        "max_epoch_f"   : 0,
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