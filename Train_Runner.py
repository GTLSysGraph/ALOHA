import argparse    
from model_zoo import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str,               default= 'Cora') #['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv','Reddit','Flickr','Yelp']
    parser.add_argument('--attack',         type=str,               default=  'no') # ['DICE-0.1','Meta_Self-0.05' ,...] 攻击方式-扰动率
    parser.add_argument('--model_name',     type = str,             default = 'MaskGAE')
    parser.add_argument('--task',           type = str,             default = 'node') # 'link' 'graph'
    parser.add_argument('--mode',           type = str,             default = 'tranductive') # 'link' 'graph'
    margs = parser.parse_args()

    if margs.model_name == 'GRACE':
        Train_GRACE_nodecls(margs)
    elif margs.model_name == 'S2GAE':
        Train_S2GAE_nodecls(margs)
    elif margs.model_name == 'MaskGAE':
        if margs.task == 'graph':
            Train_MaskGAE_graphcls(margs)
        elif margs.task == 'node':
            Train_MaskGAE_nodecls(margs)
    elif margs.model_name == 'GraphMAE':
        if margs.task == 'graph':
            Train_GraphMAE_graphcls(margs)
        elif margs.task == 'node':
            Train_GraphMAE_nodecls(margs)
    elif margs.model_name == 'DGL_VGAE':
        assert margs.task in ['node','link']
        if   margs.task == 'link':
            Train_VGAE_linkcls(margs)
        elif margs.task == 'node':
            Train_VGAE_nodecls(margs)
    elif margs.model_name in ['GAE' ,'VGAE']:
        assert margs.task in ['node','link']
        Train_GAE_cls(margs)
    elif margs.model_name == 'STABLE':
        Train_STABLE_nodecls(margs)
    elif margs.model_name == 'RobustGAE':
        Train_RobustGAE_nodecls(margs)
    elif margs.model_name == 'DiffMGAE':
        if margs.task == 'graph':
            Train_DiffMGAE_graphcls(margs)
        elif margs.task == 'node':
            Train_DiffMGAE_nodecls(margs)
    elif margs.model_name == 'DGI':
        Train_DGI_nodecls(margs)
    elif margs.model_name == 'BGRL':
        Train_BGRL_nodecls(margs)
    elif margs.model_name == 'GIN':
        assert margs.task == 'graph'
        Train_GIN_graphcls(margs)
    elif margs.model_name == 'InfoGraph':
        assert margs.task == 'graph'
        Train_InfoGraph_graphcls(margs)
    elif margs.model_name == 'MVGRL':
        assert margs.task in ['graph','node']
        if margs.task == 'graph':
            Train_MVGRL_graphcls(margs)
        elif margs.task == 'node':
            Train_MVGRL_nodecls(margs)
    elif margs.model_name == 'GCN':
        assert margs.task == 'node'
        Train_GCN_nodecls(margs)
    elif margs.model_name == 'GAT':
        assert margs.task == 'node'
        Train_GAT_nodecls(margs)
    elif margs.model_name == 'NASMGAE':
        if margs.task == 'graph':
            Train_NASMGAE_graphcls(margs)
        elif margs.task == 'node':
            Train_NASMGAE_nodecls(margs)
    else:
        raise Exception('Model not realise, wait...')
