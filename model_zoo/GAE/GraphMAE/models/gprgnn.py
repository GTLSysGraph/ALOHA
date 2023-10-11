import torch
import torch as th
import torch.nn as nn
from   torch.nn import init
import torch.nn.functional as F
import numpy as np
from   dgl.base import DGLError
import dgl.function as fn
from dgl.utils import expand_as_pair

class ada_prop(nn.Module):
    def __init__(self, P, coe, norm='both',bias=True, **kwargs):
        super(ada_prop, self).__init__()
        self.P = P
        self.coe = coe
        coes = coe*(1-coe)**np.arange(P+1)
        coes[-1] = (1-coe)**P
        self.coes = nn.Parameter(torch.tensor(coes))

        self.agg_layers = nn.ModuleList()
        for _ in range(self.P):
            self.agg_layers.append(AGG_OP(norm))


    def reset_parameters(self):
        nn.init.zeros_(self.coes)
        for p in range(self.P+1):
            self.coes.data[p] = self.coe*(1-self.coe)**p
        self.coes.data[-1] = (1-self.coe)**self.P
    
    def forward(self, g, x, edge_weight=None):
        hidden_list = []
        hidden = x*(self.coes[0])
        hidden_list.append(hidden)
        for p in range(self.P):
            x = self.agg_layers[p](g, x, edge_weight=edge_weight)
            c = self.coes[p+1]
            hidden_list.append(c*x)
            hidden = hidden + c*x
        return hidden,hidden_list

    
class AGG_OP(nn.Module):
    def __init__(self, norm):
        super(AGG_OP, self).__init__()
        self._norm = norm

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1) 
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]
          
            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)   
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            return rst



class GPRGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=.5, coe=.5, P=5):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.prop = ada_prop(P, coe)

        self.dropout = dropout
        self.head = nn.Identity()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, g, x, edge_weight=None,return_hidden=False):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)


        x = F.dropout(x, p=self.dropout, training=self.training)
        x, hidden_list = self.prop(g, x,edge_weight=edge_weight)

        if return_hidden:
            return self.head(x), hidden_list
        else:
            return self.head(x)



    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)