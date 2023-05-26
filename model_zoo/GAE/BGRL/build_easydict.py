from     easydict        import EasyDict

def build_easydict_nodecls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'BGRL'
    MDT.MODEL.PARAM = {
        'model_seed': 1,
        'data_seed': 1,
        'num_eval_splits': 3,
        'graph_encoder_layer': [512, 256, 128],
        'predictor_hidden_size': 512,
        'epochs': 10000,
        'steps': 500,
        'lr': 1e-5,
        'weight_decay': 1e-5,
        'mm': 0.99,
        'lr_warmup_epochs': 1000,
        'lr_warmup_steps': 1000,
        'drop_edge_p_1': 0.,
        'drop_feat_p_1': 0.,
        'drop_edge_p_2': 0.,
        'drop_feat_p_2': 0.,
        'logdir': "/home/songsh/GCL/model_zoo/GAE/BGRL/logdir/",
        'eval_epochs': 5,
        'use_cfg':True,
        'ckpt_path' : None,
        'batch_size': 22,
        'num_workers': 1,
        'eval_steps': 2000
    }

    return MDT