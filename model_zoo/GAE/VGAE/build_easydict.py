from easydict import EasyDict

# 节点分类任务主要是"lr_f" 和"weight_decay_f"的设置，非常关键！！！！！！！！！！！！

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'VGAE'
    MDT.MODEL.PARAM = {
        "learning_rate":0.01, # help="Initial learning rate."
        "epochs":200, #help="Number of epochs to train."      default 200
        "hidden1":512, #help="Number of units in hidden layer 1.", default 32
        "hidden2":8, #help="Number of units in hidden layer 2.", default 16
    }
    MDT.MODEL.NODECLS = {
        "max_epoch_f"   : 300,
        "lr_f"          : 0.01,
        "weight_decay_f": 0.0,
        "linear_prob"   : True,
    }

    return MDT


