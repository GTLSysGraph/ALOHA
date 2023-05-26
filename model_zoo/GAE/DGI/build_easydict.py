from easydict import EasyDict

def build_easydict_nodecls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'DGI'
    MDT.MODEL.PARAM = {
        'n_hidden' : 512,
        'n_layers' : 1,
        'dropout'  : 0.0,
        'weight_decay': 0.0,
        'patience': 20,
        'n_dgi_epochs': 300,
        'n_classifier_epochs': 300,
        'dgi_lr'   : 1e-3,
        'classifier_lr': 1e-2,
    }
    return MDT