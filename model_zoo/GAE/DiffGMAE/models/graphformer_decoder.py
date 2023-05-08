import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from ..utils import ActivationFunctionSelector

class GraphDecoder(nn.Module):
    def __init__(self,
                in_dim: int,
                embedding_dim,
                num_heads:int,
                out_dim:int,
                last_state_only:bool,
                cross_attention_input_dim:list,
                num_decoder_layers: int
                ):
        super(GraphDecoder, self).__init__()

        self.last_state_only = last_state_only

        self.in_dim_to_emb_dim = nn.Linear(in_dim, embedding_dim)

        self.graph_decoder = GraphFormerDecoder(
                                embedding_dim,
                                num_heads,
                                cross_attention_input_dim,
                                num_decoder_layers
                            )
        self.rebuild_in_dim = nn.Linear(embedding_dim,out_dim)

    def forward(self, noise_feature, resist_enc_embed,self_mask,cross_mask):
        noise_dec_feat = self.in_dim_to_emb_dim(noise_feature)
        inner_states = self.graph_decoder(noise_dec_feat, resist_enc_embed ,self_mask, cross_mask,self.last_state_only)
        x = inner_states[-1]
        rebuild_x = self.rebuild_in_dim(x)
        return rebuild_x
    

class GraphFormerDecoder(nn.Module):
    def __init__(self,
                embedding_dim: int,
                num_heads:int,
                cross_attention_input_dim:list,
                num_decoder_layers: int,
                ):
        super(GraphFormerDecoder, self).__init__()

        # self.graph_node_feature = GraphNodeFeature(
        #     num_in_degree=num_in_degree,
        #     num_out_degree=num_out_degree,
        #     hidden_dim=embedding_dim,
        # )

        self.layers = nn.ModuleList([])
        # 改不同的使用encoder的cross方式主要是改变不同layer的cross_attention_input_dim
        self.layers.extend(
            [
                GraphormerGraphDecoderLayer(
                    embedding_dim,
                    num_heads,
                    cross_attention_input_dim
                )
                for _ in range(num_decoder_layers)
            ]
        )


    def forward(self ,noise_dec_feat, resist_enc_embed, self_mask, cross_mask, last_state_only: bool = False):
        # x = self.graph_node_feature(graph, mask_nodes, noise_dec_feat)
        x = noise_dec_feat

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x, resist_enc_embed, self_mask, cross_mask
            )
            if not last_state_only:
                inner_states.append(x)

        
        if last_state_only:
            inner_states = [x]
        return inner_states



class GraphormerGraphDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        cross_attention_input_dim,
        activation_fn: str = "relu",
        
    ):
        super().__init__()

        self.cross_attention_input_dim = cross_attention_input_dim
        self.activation_fn  = ActivationFunctionSelector(activation_fn)
        self.self_attn       = MultiHeadAttention(d_model,num_heads,[])
        self.cross_attn      = MultiHeadAttention(d_model,num_heads,cross_attention_input_dim)

    def forward(self, noise_x, resist_x, self_mask, cross_mask, use_residual = False):
        if use_residual:
            residual = noise_x
        
        attn_final = []
        if len(self.cross_attention_input_dim) == 0:
            # self attention
            x, attn = self.self_attn(noise_x,noise_x,noise_x,self_mask)
            attn_final.append(attn)
        else:
            # cross attention 谁提供Q，最后输出的就是谁的embedding
            x_cross, attn_cross = self.cross_attn(noise_x,resist_x,resist_x,cross_mask)
            attn_final.append(attn_cross)
            # self attention
            x_self,  attn_self = self.self_attn(noise_x,noise_x,noise_x,self_mask)
            attn_final.append(attn_self)

            # 这里对self attention和 cross_attention进行融合
            x =  x_cross
            

        # x = self.dropout(x)
        # x = self.activation_fn(x)
        
        if use_residual:
            x = residual + x
        return x, attn_final


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if len(d_input) == 0  :
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input
        # Embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads
        # These are still of dimension d_model. To split into number of heads
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)
        # Outputs of all sub-layers need to be of dimension d_model
        # self.W_h = nn.Linear(d_model, d_model)

        #### 不用Q和K相乘得到注意力的方式，模仿gat里面
        self.attn_q = nn.Parameter(torch.FloatTensor(size=(self.num_heads, 1, self.d_k)))
        self.attn_k = nn.Parameter(torch.FloatTensor(size=(self.num_heads, 1, self.d_k)))

        self.leaky_relu = nn.LeakyReLU(0.2)
    
        self.reset_parameters()
    
    def reset_parameters(self):
       
        torch.nn.init.xavier_normal_(self.attn_q, gain=1)
        torch.nn.init.xavier_normal_(self.attn_k, gain=1)


    def scaled_dot_product_attention(self, Q, K, V, mask):
        # Scaling by d_k so that the soft(arg)max doesnt saturate
        Q = Q / np.sqrt(self.d_k)  # (n_heads, q_length, dim_per_head)
        # 仿照gat的score获取方法
        q_attn = (Q * self.attn_q).sum(dim=-1).unsqueeze(-1)
        k_attn = (K * self.attn_k).sum(dim=-1).unsqueeze(-1)
        q_attn = q_attn.unsqueeze(-2).expand(-1,-1,K.shape[-2],-1)
        scores = (q_attn + k_attn).squeeze(-1)
        scores = self.leaky_relu(scores * mask)
        # self attention的score方法，效果很差
        # scores = self.leaky_relu(torch.matmul(Q, K.transpose(1,2))) # (n_heads, q_length, k_length)


        A =F.softmax(scores,dim=-1)  # (n_heads, q_length, k_length)
        # Get the weighted average of the values
        H = torch.matmul(A, V)  # (n_heads, q_length, dim_per_head)
        return H, A
    
    def split_heads(self, x):
        return x.view(-1, self.num_heads, self.d_k).transpose(0, 1)

    def group_heads(self, x):
        # (n_heads, q_length, k_length) -> (q_length, n_heads, k_length)
        return x.transpose(0, 1).contiguous().view(-1, self.num_heads * self.d_k)


    def forward(self, X_q, X_k, X_v, mask):
        # After transforming, split into num_heads
        Q = self.split_heads(self.W_q(X_q))
        K = self.split_heads(self.W_k(X_k))
        V = self.split_heads(self.W_v(X_v))
        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V, mask)
        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat)  # (q_length, dim)
        # Final linear layer
        # H = self.W_h(H_cat)  # (q_length, dim)
        return H_cat, A




# class GraphNodeFeature(nn.Module):
#     """
#     Compute node features for each node in the graph.
#     """
#     def __init__(
#         self, num_in_degree, num_out_degree, hidden_dim
#     ):
#         super(GraphNodeFeature, self).__init__()

#         self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim)
#         self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim)


#     def forward(self, graph, mask_nodes, x):
#         node_feature = (
#             x
#             + self.in_degree_encoder(graph.in_degrees())[mask_nodes]
#             + self.out_degree_encoder(graph.out_degrees())[mask_nodes]
#         )
    
#         return node_feature
