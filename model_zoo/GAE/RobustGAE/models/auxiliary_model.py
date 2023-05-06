import torch.nn as nn
import torch
import torch.nn.functional as F
from ..utils import gumbel_softmax

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")



class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class EdgeReEnhence(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=2, mode= 'concat',num_layers=1, dropout=0.5,activation='relu'
    ):
        super().__init__()
        self.mode = mode
        self.mlps = nn.ModuleList()

        if self.mode == 'concat':
            for i in range(num_layers):
                first_channels = in_channels * 2 if i == 0 else hidden_channels
                second_channels = out_channels if i == num_layers - 1 else hidden_channels
                self.mlps.append(nn.Linear(first_channels, second_channels))
        elif self.mode == 'sim':
            for i in range(num_layers):
                first_channels = in_channels if i == 0 else hidden_channels
                second_channels = out_channels if i == num_layers - 1 else hidden_channels
                self.mlps.append(nn.Linear(first_channels, second_channels))
        elif self.mode == 'cos':
            pass
        else:
            raise Exception('Unknown EdgeReEnhence Mode!')
        
        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)


    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, use_gumbel_softmax = True,sigmoid=True):
        if self.mode == 'sim':
            x = z[edge[0]] * z[edge[1]]
            for i, mlp in enumerate(self.mlps[:-1]):
                x = self.dropout(x)
                x = mlp(x)
                x = self.activation(x)
            x = self.mlps[-1](x)

        elif self.mode == 'concat':
            x = torch.cat((z[edge[0]],z[edge[1]]),dim = 1)
            for i, mlp in enumerate(self.mlps[:-1]):
                x = self.dropout(x)
                x = mlp(x)
                x = self.activation(x)
            x = self.mlps[-1](x)

        elif self.mode == 'cos':
            z = z / torch.norm(z,dim=-1, keepdim=True)
            z_cos = torch.mm(z,z.t())
            readd_edges_index = torch.nonzero(z_cos[edge[0],edge[1]] > 0.95).squeeze()
            perb_edge_cos =  z_cos[edge[0],edge[1]].unsqueeze(-1)
            x = torch.cat((perb_edge_cos, 1-perb_edge_cos),dim = -1)           

        if sigmoid:
            x = x.sigmoid()

        if use_gumbel_softmax:
            readd_edges_mask = gumbel_softmax(x, temperature=0.5, hard=True)
            # 第一列是需要添加的
            readd_edges_mask = readd_edges_mask[..., 0]
        else:
            # 第一列表示添加，第二列不添加
            readd_edges_mask = 1 - F.softmax(x,dim = 1).argmax(dim=1)

        if self.mode == 'cos':
            # 添加的边严格要求一下，从大于某cos值的边里面选，这些边根据概率gumbel之后第二次选择，这样的边添加到refine当中
            readd_edges_index = readd_edges_index[(readd_edges_mask == 1)[readd_edges_index]]
        else:
            readd_edges_index = readd_edges_mask.nonzero().squeeze(-1)

        return readd_edges_index




class StructDecoder(nn.Module):
    """Simple MLP Struct Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=40,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, x,  sigmoid=True, reduction=False):
    
        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        x = x / torch.norm(x,dim=-1, keepdim=True)
        z = torch.mm(x,x.t())

        if sigmoid:
            return z.sigmoid()
        else:
            return z