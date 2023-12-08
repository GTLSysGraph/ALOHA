from easydict import EasyDict

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'GraphMAE2'
    MDT.MODEL.PARAM = {
        'seeds'         : [1,2,3,4], # 0.8367±0.0050
        "device": 0,
        "max_epoch":500,           
        "warmup_steps":-1,
        "num_heads":4,                  
        "num_out_heads":1,                 
        "num_layers":2,              
        "num_dec_layers":1,
        "num_remasking":3,
        "num_hidden":512,                  
        "residual":False,                 
        "in_drop":.2,                       
        "attn_drop":.1,                 
        "norm":None,
        "lr":0.001,                    
        "weight_decay":0,               
        "negative_slope":0.2,
        "activation":"prelu",
        "mask_rate":0.5,
        "remask_rate":0.5,
        "remask_method":"random",
        "mask_type":"mask",
        "mask_method":"random",
        "drop_edge_rate": 0.0,
        "drop_edge_rate_f":0.0,
        "encoder":"gat",
        "decoder":"gat",
        "loss_fn":"sce",
        "alpha_l":2,
        "optimizer":"adam",
        "max_epoch_f":300,
        "lr_f":0.01,
        "weight_decay_f":0.0,
        "linear_prob":False,
        "no_pretrain":False,
        "load_model":False,
        "checkpoint_path":None,
        "use_cfg":True,
        "logging":False,
        "scheduler":False,
        "batch_size":256,
        "batch_size_f":128,
        "sampling_method":"saint",
        "label_rate":1.0,
        "ego_graph_file_path":None,
        "data_dir":"data",
        "lam":1.0,
        "full_graph_forward":False,
        "delayed_ema_epoch":0,
        "replace_rate":0.0,
        "momentum":0.996,
        }

    return MDT


