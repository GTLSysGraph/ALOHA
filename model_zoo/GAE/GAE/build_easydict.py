from easydict import EasyDict
import yaml

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = '(V)GAE'
    MDT.MODEL.PARAM = {
        'linear': False,
        'epochs': 400,
        'variational':True,
        'seed': 1
    }

    MDT.MODEL.LP_PARAM = {
        'runs': 10,
        'nodeclas_weight_decay': 1e-3,
        'debug': False
    }
    return MDT