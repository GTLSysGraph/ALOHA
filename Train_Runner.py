import argparse    
from model_zoo import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=  str,               default= 'Cora') #['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv','Reddit','Flickr','Yelp']
    parser.add_argument('--attack',         type=  str,               default=  'no') # ['DICE-0.1','Meta_Self-0.05' ,...] 攻击方式-扰动率
    parser.add_argument('--model_name',     type = str,               default = 'MaskGAE')
    parser.add_argument('--task',           type = str,               default = 'nodecls') # 'linkcls' 'graphcls'
    parser.add_argument('--mode',           type = str,               default = 'tranductive') # inductive mini-batch
    


    # unit test data
    parser.add_argument('--scenario',               type = str,                 default = 'poisoning')  #"evasion"
    parser.add_argument('--adaptive_attack_model',  type = str,                 default = 'jaccard_gcn') # "clean, gcn", "jaccard_gcn", "svd_gcn", "rgcn", "pro_gnn", "gnn_guard", "grand", "soft_median_gdc"
    parser.add_argument('--split',                  type = str,                 default = 0 )
        # 这两个值自己取文件夹中对应去看
    parser.add_argument('--budget',                 type=  str,                 default=5)                
    parser.add_argument('--unit_ptb',               type=  str,                 default= 0.0,    help='unit rate.')
    
    margs = parser.parse_args()



    Train_func = 'Train_' + margs.model_name + '_' + margs.task
    eval(Train_func)(margs)

    # if margs.model_name == 'GRACE':
    #     Train_GRACE_nodecls(margs)
    # elif margs.model_name == 'S2GAE':
    #     Train_S2GAE_nodecls(margs)
    # elif margs.model_name == 'MaskGAE':
    #     if margs.task == 'graph':
    #         Train_MaskGAE_graphcls(margs)
    #     elif margs.task == 'node':
    #         Train_MaskGAE_nodecls(margs)
    # elif margs.model_name == 'GraphMAE':
    #     if margs.task == 'graph':
    #         Train_GraphMAE_graphcls(margs)
    #     elif margs.task == 'node':
    #         Train_GraphMAE_nodecls(margs)
    # elif margs.model_name == 'DGL_VGAE':
    #     assert margs.task in ['node','link']
    #     if   margs.task == 'link':
    #         Train_VGAE_linkcls(margs)
    #     elif margs.task == 'node':
    #         Train_VGAE_nodecls(margs)
    # elif margs.model_name in ['GAE' ,'VGAE']:
    #     assert margs.task in ['node','link']
    #     Train_GAE_cls(margs)
    # elif margs.model_name == 'STABLE':
    #     Train_STABLE_nodecls(margs)
    # elif margs.model_name == 'RobustGAE':
    #     Train_RobustGAE_nodecls(margs)
    # elif margs.model_name == 'DiffMGAE':
    #     if margs.task == 'graph':
    #         Train_DiffMGAE_graphcls(margs)
    #     elif margs.task == 'node':
    #         Train_DiffMGAE_nodecls(margs)
    # elif margs.model_name == 'DGI':
    #     Train_DGI_nodecls(margs)
    # elif margs.model_name == 'BGRL':
    #     Train_BGRL_nodecls(margs)
    # elif margs.model_name == 'GIN':
    #     assert margs.task == 'graph'
    #     Train_GIN_graphcls(margs)
    # elif margs.model_name == 'InfoGraph':
    #     assert margs.task == 'graph'
    #     Train_InfoGraph_graphcls(margs)
    # elif margs.model_name == 'MVGRL':
    #     assert margs.task in ['graph','node']
    #     if margs.task == 'graph':
    #         Train_MVGRL_graphcls(margs)
    #     elif margs.task == 'node':
    #         Train_MVGRL_nodecls(margs)
    # elif margs.model_name == 'GCN':
    #     assert margs.task == 'node'
    #     Train_GCN_nodecls(margs)
    # elif margs.model_name == 'GAT':
    #     assert margs.task == 'node'
    #     Train_GAT_nodecls(margs)
    # elif margs.model_name == 'SPMGAE':
    #     if margs.task == 'graph':
    #         Train_SPMGAE_graphcls(margs)
    #     elif margs.task == 'node':
    #         Train_SPMGAE_nodecls(margs)
    # else:
    #     raise Exception('Model not realise, wait...')
