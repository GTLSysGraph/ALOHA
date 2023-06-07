
from easydict import EasyDict

def build_easydict():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'GAT'
    MDT.MODEL.PARAM = {
        'seeds': [2023,2024,2025],
        'num_layers':2,
        'n_hidden': 8,
        'heads' : [8,8]
    }

    return MDT

# ppi                      256   [4,4,6]  three-layer   400 epoch :  0.9730±0.002        注意所有的feat_drop=0., attn_drop=0. 论文里提到了The training sets for this task are sufficiently large and we found no need to apply L2 regularization or dropout 

# Cora                     8     [8,8]    two-layer     200 epoch :  0.8330±0.0028
# Citeseer                 8     [8,4]    two-layer     200 epoch :  0.7073±0.0076
# Pubmed                   8     [8,1]    two-layer     200 epoch :  0.7740±0.0043
# ogbn-arxiv_undirected    8     [8,8]    two-layer     200 epoch :  0.7009±0.0001       feat_drop=0., attn_drop=0.
# reddit                   8     [8,8]    two-layer     200 epoch :  0.9584±0.0057       feat_drop=0., attn_drop=0.