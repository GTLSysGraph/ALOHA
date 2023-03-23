import argparse    
import torch
from model_zoo import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str,               default= 'Cora') #['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv','Reddit','Flickr','Yelp']
    parser.add_argument('--attack',         type=str,               default=  'no') # ['DICE-0.1','Meta_Self-0.05' ,...] 攻击方式-扰动率
    parser.add_argument('--gpu_id',         type=int,               default=  1) 
    parser.add_argument('--model_name',     type = str,             default = 'MaskGAE')
    parser.add_argument('--task',           type = str,             default = 'node') # 'link' 'graph'
    margs = parser.parse_args()

    if margs.model_name == 'GRACE':
        Train_Grace(margs)
    elif margs.model_name == 'S2GAE':
        Train_S2GAE_nodecls(margs)
    elif margs.model_name == 'MaskGAE':
        Train_MaskGAE_nodecls(margs)
    elif margs.model_name == 'GraphMAE':
        Train_GraphMAE_nodecls(margs)
    elif margs.model_name == 'VGAE':
        if margs.task == 'link':
            Train_VGAE_linkcls(margs)
        elif margs.task == 'node':
            Train_VGAE_nodecls(margs)

    
