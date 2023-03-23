from model_zoo.GCL.GRACE.Train_Grace    import Train_Grace
from model_zoo.GAE.S2GAE.Train_S2GAE_nodecls import Train_S2GAE_nodecls
from model_zoo.GAE.MaskGAE.Train_MaskGAE_nodecls import Train_MaskGAE_nodecls
from model_zoo.GAE.GraphMAE.Train_GraphMAE_nodecls import Train_GraphMAE_nodecls
from model_zoo.GAE.VGAE.Train_VGAE_linkcls import Train_VGAE_linkcls
from model_zoo.GAE.VGAE.Train_VGAE_nodecls import Train_VGAE_nodecls

__all__ = ['Train_Grace','Train_S2GAE_nodecls','Train_MaskGAE_nodecls','Train_GraphMAE_nodecls','Train_VGAE_linkcls','Train_VGAE_nodecls']