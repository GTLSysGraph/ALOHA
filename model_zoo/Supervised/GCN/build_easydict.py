
from easydict import EasyDict

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'GCN'
    MDT.MODEL.PARAM = {
        'seeds': [2023,2024,2025],
        'n_hidden': 64
    }

    return MDT

# ppi                      8192  400 epoch :  0.7714±0.0063
# Cora                     16    200 epoch :  0.8160±0.0014
# Citeseer                 16    200 epoch :  0.7090±0.0014
# Pubmed                   512   200 epoch :  0.7947±0.0012
# ogbn-arxiv_undirected    1024  200 epoch :  0.7094±0.0017
# reddit                   64    200 epoch :  0.9501±0.0008