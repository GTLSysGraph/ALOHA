import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F



class GAT(nn.Module):
    def __init__(self,num_layers, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, heads[0], feat_drop=0., attn_drop=0., activation=F.elu)
        )
        # hidden layers
        l = 0 # 如果只有两层，初始化为0
        for l in range(1, num_layers - 1): # 2层这里啥也没有
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                dglnn.GATConv(
                    hid_size * heads[l - 1],
                    hid_size,
                    heads[l],
                    feat_drop=0.,
                    attn_drop=0.,
                    residual=False,
                    activation=F.elu,
                )
            )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[l],
                out_size,
                heads[l+1],
                feat_drop=0.,
                attn_drop=0.,
                residual=False,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 2:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


