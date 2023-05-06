import torch.nn as nn
import torch
import math
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_timestep):
        super(PositionalEncoding, self).__init__()
        self.d_timestep = d_timestep
    
        
    def forward(self, time):
        len = time.shape[0]
        self.d_timestep_new = self.d_timestep // 2 * 2
        pe = torch.zeros(len, self.d_timestep_new) 
        div_term = torch.exp(torch.arange(0, self.d_timestep_new, 2) *
                        -(math.log(10000.0) / self.d_timestep_new)).to(time.device)
        embeddings = time[:, None] * div_term[None, :]
        pe[:, 0::2] = torch.sin(embeddings)
        pe[:, 1::2] = torch.cos(embeddings)
        if self.d_timestep % 2:
                pe = torch.cat([pe, torch.zeros_like(pe[:, :1])], dim=-1)
        return pe