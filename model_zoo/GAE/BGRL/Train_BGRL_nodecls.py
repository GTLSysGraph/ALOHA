from .Train_BGRL_nodecls_tranductive import Train_BGRL_nodecls_tranductive
from .Train_BGRL_nodecls_inductive   import Train_BGRL_nodecls_inductive

def Train_BGRL_nodecls(margs):
    if margs.mode == 'tranductive':
        Train_BGRL_nodecls_tranductive(margs)
    elif margs.mode == 'inductive':
        Train_BGRL_nodecls_inductive(margs)
    else:
        raise Exception('Unknown mode!')
    