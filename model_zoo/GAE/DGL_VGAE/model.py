import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv


class VGAEModel(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim,device):
        super(VGAEModel, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim

        layers = [
            GraphConv(
                self.in_dim,
                self.hidden1_dim,
                activation=F.relu,
                allow_zero_in_degree=True,
            ),
            GraphConv(
                self.hidden1_dim,
                self.hidden2_dim,
                activation=lambda x: x,
                allow_zero_in_degree=True,
            ),
            GraphConv(
                self.hidden1_dim,
                self.hidden2_dim,
                activation=lambda x: x,
                allow_zero_in_degree=True,
            ),
        ]
        self.layers = nn.ModuleList(layers)

        self.head = nn.Identity()


    def encoder(self, g, features):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(self.device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(self.device)
        return sampled_z

    def decoder(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec

    def forward(self, g, features):
        z = self.encoder(g, features)
        adj_rec = self.decoder(z)
        return adj_rec

    # 自己参考GraphMAE修改添加
    def embed(self, g, features):
        rep = self.encoder(g, features)
        return self.head(rep)
    
    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.hidden2_dim, num_classes)