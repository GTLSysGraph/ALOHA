from easydict import EasyDict

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'VGAE'
    MDT.MODEL.PARAM = {
        "learning_rate":0.01, # help="Initial learning rate."
        "epochs":200, #help="Number of epochs to train."
        "hidden1":32, #help="Number of units in hidden layer 1.",
        "hidden2":16, #help="Number of units in hidden layer 2.",
    }
    MDT.MODEL.NODECLS = {
        "max_epoch_f"   : 300,
        "lr_f"          : 0.01,
        "weight_decay_f": 1e-4,
        "linear_prob"   : True,
    }

    return MDT


