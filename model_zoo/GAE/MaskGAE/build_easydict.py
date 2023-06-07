from easydict import EasyDict
import yaml
import logging

def load_best_configs(param, dataset_name, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if dataset_name not in configs:
        logging.info("Best args not found")
        return param

    logging.info("Using best configs")
    configs = configs[dataset_name]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(param, k, v)
    print("------ Use best configs ------")
    return param



def build_easydict_nodecls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'MaskGAE'
    MDT.MODEL.PARAM = {
        "dataset": "Cora", #help="Datasets. (default: Cora)")
        "mask":"Path", #help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
        'seed':2022, #help='Random seed for model and dataset. (default: 2022)')

        "layer":"gcn", #help="GNN layer, (default: gcn)")
        "encoder_activation":"elu", #help="Activation function for GNN encoder, (default: elu)")
        'encoder_channels':128, #help='Channels of GNN encoder layers. (default: 128)')
        'hidden_channels':64, #help='Channels of hidden representation. (default: 64)')
        'decoder_channels':32, #help='Channels of decoder layers. (default: 128)')
        'encoder_layers':2, #help='Number of layers for encoder. (default: 2)')
        'decoder_layers':2,# help='Number of layers for decoders. (default: 2)')
        'encoder_dropout':0.8,# help='Dropout probability of encoder. (default: 0.8)')
        'decoder_dropout':0.2, #help='Dropout probability of decoder. (default: 0.2)')
        'alpha':0.001, #help='loss weight for degree prediction. (default: 0.)')

        'lr':0.02, #help='Learning rate for training. (default: 0.01)')
        'weight_decay':5e-5, #help='weight_decay for link prediction training. (default: 5e-5)')
        'grad_norm':1.0, #help='grad_norm for training. (default: 1.0.)')
        'batch_size':2**16, #help='Number of batch size for link prediction training. (default: 2**16)')

        "start":"node", #help="Which Type to sample starting nodes for random walks, (default: node)")
        'p':0.7, #help='Mask ratio or sample ratio for MaskEdge/MaskPath')

        'bn':True,# help='Whether to use batch normalization for GNN encoder. (default: False)')
        'l2_normalize':True, #help='Whether to use l2 normalize output embedding. (default: False)')
        'nodeclas_weight_decay':0.1,# help='weight_decay for node classification training. (default: 1e-3)')

        'epochs':500, #help='Number of training epochs. (default: 500)')
        'runs': 10, #help='Number of runs. (default: 10)')
        'eval_period':30, #help='(default: 30)')
        'patience':30, #help='(default: 30)')
        "debug": False,
        # "save_path":"model_nodeclas", #help="save path for model. (default: model_nodeclas)")

    }
    return MDT


def build_easydict_graphcls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'MaskGAE'
    MDT.MODEL.PARAM = {
        "mask":"Path", #help="Masking stractegy, `Path`, `Edge` or `None` (default: Path)")
        'seed':2022, #help='Random seed for model and dataset. (default: 2022)')

        "layer":"gin", #help="GNN layer, (default: gcn)")
        "encoder_activation":"elu", #help="Activation function for GNN encoder, (default: elu)")
        'encoder_channels':128, #help='Channels of GNN encoder layers. (default: 128)')
        'hidden_channels':64, #help='Channels of hidden representation. (default: 64)')
        'decoder_channels':32, #help='Channels of decoder layers. (default: 128)')
        'encoder_layers':2, #help='Number of layers for encoder. (default: 2)')
        'decoder_layers':2,# help='Number of layers for decoders. (default: 2)')
        'encoder_dropout':0.8,# help='Dropout probability of encoder. (default: 0.8)')
        'decoder_dropout':0.2, #help='Dropout probability of decoder. (default: 0.2)')
        'alpha':0., #help='loss weight for degree prediction. (default: 0.)')

        'lr':0.01, #help='Learning rate for training. (default: 0.01)')
        'weight_decay':5e-5, #help='weight_decay for link prediction training. (default: 5e-5)')
        'grad_norm':1.0, #help='grad_norm for training. (default: 1.0.)')
        'batch_size':2**16, #help='Number of batch size for link prediction training. (default: 2**16)')

        "start":"node", #help="Which Type to sample starting nodes for random walks, (default: node)")
        'p':0.7, #help='Mask ratio or sample ratio for MaskEdge/MaskPath')

        'bn':True,# help='Whether to use batch normalization for GNN encoder. (default: False)')
        'l2_normalize':True, #help='Whether to use l2 normalize output embedding. (default: False)')
        'graphclas_weight_decay':1e-3,# help='weight_decay for node classification training. (default: 1e-3)')
        'pooling': 'mean',
        'graph_batch_size': 256,

        'epochs':300, #help='Number of training epochs. (default: 500)')
        'runs': 10, #help='Number of runs. (default: 10)')
        'eval_period':10, #help='(default: 10)')
        'patience':10, #help='(default: 10)')
        "debug": False,
        "save_path": "model_graphclas", # "model_graphclas", help="save path for model. (default: model_graphclas)")
    }
    return MDT