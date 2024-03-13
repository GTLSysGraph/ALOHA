# from model_zoo.GCL.GRACE.Train_GRACE_nodecls    import Train_GRACE_nodecls
from model_zoo.GCL.STABLE.Train_STABLE_nodecls    import Train_STABLE_nodecls
from model_zoo.GCL.InfoGraph.Train_InfoGraph_graphcls import Train_InfoGraph_graphcls
from model_zoo.GCL.MVGRL.Train_MVGRL_graphcls import Train_MVGRL_graphcls
from model_zoo.GCL.MVGRL.Train_MVGRL_nodecls import Train_MVGRL_nodecls



# from model_zoo.GAE.S2GAE.Train_S2GAE_nodecls import Train_S2GAE_nodecls
# from model_zoo.GAE.MaskGAE.Train_MaskGAE_nodecls import Train_MaskGAE_nodecls
# from model_zoo.GAE.MaskGAE.Train_MaskGAE_graphcls import Train_MaskGAE_graphcls
# from model_zoo.GAE.GAE.Train_GAE_cls import Train_GAE_cls
# from model_zoo.GAE.BGRL.Train_BGRL_nodecls import Train_BGRL_nodecls



from model_zoo.GAE.DGL_VGAE.Train_VGAEDGL_linkcls import Train_VGAE_linkcls
from model_zoo.GAE.DGL_VGAE.Train_VGAEDGL_nodecls import Train_VGAE_nodecls
from model_zoo.GAE.SPMGAE.Train_SPMGAE_nodecls  import Train_SPMGAE_nodecls
from model_zoo.GAE.SPMGAE.Train_SPMGAE_graphcls import Train_SPMGAE_graphcls
from model_zoo.GAE.GraphMAE.Train_GraphMAE_nodecls import Train_GraphMAE_nodecls
from model_zoo.GAE.GraphMAE.Train_GraphMAE_graphcls import Train_GraphMAE_graphcls
from model_zoo.GAE.GraphMAE2.Train_GraphMAE2_nodecls import Train_GraphMAE2_nodecls
from model_zoo.GAE.RobustGAE.Train_RobustGAE_nodecls import Train_RobustGAE_nodecls
from model_zoo.GAE.DiffMGAE.Train_DiffMGAE_nodecls import Train_DiffMGAE_nodecls
from model_zoo.GAE.DiffMGAE.Train_DiffMGAE_graphcls import Train_DiffMGAE_graphcls
from model_zoo.GAE.DGI.Train_DGI_nodecls import Train_DGI_nodecls

from model_zoo.Supervised.GIN.Train_GIN_graphcls import Train_GIN_graphcls
from model_zoo.Supervised.GCN.Train_GCN_nodecls import Train_GCN_nodecls
from model_zoo.Supervised.GAT.Train_GAT_nodecls import Train_GAT_nodecls

__all__ = [ 
            # dgl backend
            'Train_STABLE_nodecls',
            'Train_GraphMAE_nodecls','Train_GraphMAE_graphcls',
            'Train_GraphMAE2_nodecls',
            'Train_VGAE_linkcls','Train_VGAE_nodecls',
            'Train_RobustGAE_nodecls',
            'Train_DiffMGAE_nodecls','Train_DiffMGAE_graphcls',
            'Train_DGI_nodecls',
            'Train_GIN_graphcls',
            'Train_InfoGraph_graphcls',
            'Train_MVGRL_graphcls','Train_MVGRL_nodecls',
            'Train_GCN_nodecls',
            'Train_GAT_nodecls',
            'Train_SPMGAE_nodecls','Train_SPMGAE_graphcls',
            
            # pyg backend
            # 'Train_GRACE_nodecls',
            # 'Train_S2GAE_nodecls',
            # 'Train_MaskGAE_nodecls','Train_MaskGAE_graphcls',
            # 'Train_GAE_cls',
            # 'Train_BGRL_nodecls',
            ]

