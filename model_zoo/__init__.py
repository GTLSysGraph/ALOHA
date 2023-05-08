from model_zoo.GCL.GRACE.Train_GRACE_nodecls    import Train_GRACE_nodecls
from model_zoo.GCL.STABLE.Train_STABLE_nodecls    import Train_STABLE_nodecls

from model_zoo.GAE.S2GAE.Train_S2GAE_nodecls import Train_S2GAE_nodecls
from model_zoo.GAE.MaskGAE.Train_MaskGAE_nodecls import Train_MaskGAE_nodecls

from model_zoo.GAE.GraphMAE.Train_GraphMAE_nodecls import Train_GraphMAE_nodecls
from model_zoo.GAE.GraphMAE.Train_GraphMAE_graphcls import Train_GraphMAE_graphcls

from model_zoo.GAE.VGAE.Train_VGAE_linkcls import Train_VGAE_linkcls
from model_zoo.GAE.VGAE.Train_VGAE_nodecls import Train_VGAE_nodecls
from model_zoo.GAE.RobustGAE.Train_RobustGAE_nodecls import Train_RobustGAE_nodecls
from model_zoo.GAE.DiffGMAE.Train_DiffGMAE_nodecls import Train_DiffGMAE_nodecls

__all__ = ['Train_GRACE_nodecls',
            'Train_STABLE_nodecls',
            'Train_S2GAE_nodecls',
            'Train_MaskGAE_nodecls',
            'Train_GraphMAE_nodecls','Train_GraphMAE_graphcls',
            'Train_VGAE_linkcls','Train_VGAE_nodecls',
            'Train_RobustGAE_nodecls',
            'Train_DiffGMAE_nodecls']

