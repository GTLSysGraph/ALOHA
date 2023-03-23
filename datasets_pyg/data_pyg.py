import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon,Flickr,Reddit,Yelp
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset # 会卡住 pip uninstall setuptools 解决
from datasets_pyg.Attack_data.attackdata import AttackDataset


def get_dataset(path, name, attackmethod= None, attackptb = None):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv','Flickr','Yelp','Reddit','Attack-Cora','Attack-Citeseer','Attack-Pubmed','Attack-polblogs']
    
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('/home/songsh/GCL/datasets_pyg')


    if name == 'Coauthor-Phy':
        return Coauthor(osp.join(root_path, 'Coauthor'), name='physics', transform=T.NormalizeFeatures())

    if name == 'Coauthor-CS':
        return Coauthor(osp.join(root_path, 'Coauthor'), name='cs', transform=T.NormalizeFeatures())
      
    if name == 'Yelp':
        return Yelp(root=path, transform=T.NormalizeFeatures())

    if name == 'Flickr':
        return Flickr(root=path, transform=T.NormalizeFeatures())

    if name == 'Reddit':
        return Reddit(root=path, transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    
    if name.startswith('Attack'):
        return AttackDataset(root = path, name = name, attackmethod = attackmethod, ptb_rate=attackptb, transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())

