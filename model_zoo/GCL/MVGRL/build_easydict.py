from easydict import EasyDict

def build_MVGRL_graphcls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'MVGRL'
    MDT.MODEL.PARAM = {
        'epochs': 20,
        'lr': 0.001,
        'n_layers':4,
        'hid_dim':128,
        'batch_size': 64,
        'patience':20,
        'wd': 0.
    }
    
    return MDT


def build_MVGRL_nodecls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'MVGRL'
    MDT.MODEL.PARAM = {
        'epochs': 500,
        'hid_dim':512,
        'patience':20,
        'wd1': 0.,
        'wd2': 0.,
        'lr1':0.001,
        'lr2':0.01,
        'epsilon': 0.01,
        'sample_size':2000,
        'device':-1,
        'use_sample':False
    }
    
    return MDT