
from easydict import EasyDict

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'DiffGMAE'
    MDT.MODEL.PARAM = {
        'seeds'         : [2022,2023,2024,2025], 
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

        # add by ssh
        'remask_rate'     : 0.6,  
        'timestep'        : 10000,
        'beta_schedule'   : 'linear',
        'start_t'         : 9000,
        'lamda_loss'      : 0.1,
        'lamda_neg_ratio' : 0.0,
        'momentum'        : 0.996,

        # for graph classification
        "pooling"    : "mean",
        "deg4feat"   : False,
        "batch_size" :32
    }

    return MDT