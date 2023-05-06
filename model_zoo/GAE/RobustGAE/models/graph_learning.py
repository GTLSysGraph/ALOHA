import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import VERY_SMALL_NUMBER,INF,to_cuda


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=2, metric_type='attention'):
        super(GraphLearner, self).__init__()
        self.topk = topk
        self.epsilon = epsilon
        self.metric_type = metric_type

        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        print('[ Graph Learner metric type: {} ]'.format(metric_type))

    
    def forward(self, context, ctx_mask=None, device=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)
        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))

            attention /= len(self.linear_sims)
            markoff_value = -INF

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0

        # post process attention
        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value,device)
        return attention
    

    def build_knn_neighbourhood(self, attention, topk, markoff_value,device):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = self.to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix
    

