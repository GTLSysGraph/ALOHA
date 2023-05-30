from easydict import EasyDict

def build_InfoGraph_graphcls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'InfoGraph'
    MDT.MODEL.PARAM = {
        'epochs': 20,
        'lr': 0.01,
        'n_layers':5,
        'hid_dim':32,
        'batch_size': 128,
        'log_interval':1
    }
    
    return MDT