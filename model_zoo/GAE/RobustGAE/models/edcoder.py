import torch
import torch.nn as nn
from typing import Optional
from itertools import chain
from functools import partial
from ..utils import create_norm, drop_edge
from .loss_func import sce_loss,ce_loss,l2_loss
from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
import dgl
from .auxiliary_model import EdgeDecoder,EdgeReEnhence,StructDecoder
from ..utils import random_add_edges,to_cuda,normalize_adj, get_reliable_neighbors,kl_categorical,VERY_SMALL_NUMBER,INF
from .graph_learning import GraphLearner
import numpy as np
import torch.nn.functional as F

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod




class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            updata_guide_model = 1,
            decay = 0.1,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            
         ):
        super(PreModel, self).__init__()



        ###
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden: 
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
            dec_out_dim  =   num_hidden * num_layers
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
            dec_out_dim  =   num_hidden
    
        

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            # 本身的目的是重建特征 out dim 边embedding？试一下 或者重新定义个edgecoder
            out_dim=dec_out_dim, 
            num_layers=1, # in_dim, out_dim
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)


        ####################################################### add by ssh

        # self.fc_cat = nn.Linear(struct_num_hidden, 2)
        # self.fc_out_1 = nn.Linear(struct_num_hidden * 2, struct_num_hidden)
        # self.fc_out_2 = nn.Linear(struct_num_hidden * 2, struct_num_hidden)      
        # self._struct_type = 'gcn'
        # struct_num_hidden = 128

        # self.feat_embed   = nn.Linear(in_dim, struct_num_hidden)
        # self.struct_embed =   setup_module(
        #     m_type=self._struct_type,
        #     enc_dec="struct",
        #     in_dim=in_dim,
        #     num_hidden=struct_num_hidden,
        #     out_dim=struct_num_hidden,
        #     num_layers=num_layers,
        #     nhead=1,
        #     nhead_out=1,
        #     concat_out=True,
        #     activation=activation,
        #     dropout=feat_drop,
        #     attn_drop=attn_drop,
        #     negative_slope=negative_slope,
        #     residual=residual,
        #     norm=norm,
        # )



        # self.graph_learn = False
        # self.graph_learn_regularization = True

        # self.eps_adj = 4e-5
        # self.max_iter = 10
        # self.graph_learn_ratio = 0
        # self.smoothness_ratio =  0.2
        # self.degree_ratio =  0
        # self.sparsity_ratio = 0

        # graph_learn_hidden_size = 70
        # graph_learn_epsilon= 0 # weighted_cosine: 0
        # graph_learn_topk = None # 200
        # graph_learn_num_pers= 4 # weighted_cosine: GL: 4, IGL: 4
        # self.graph_metric_type = 'attention'

        # if self.graph_learn and self.graph_learn_regularization:
        #     graph_learn_func = GraphLearner
        #     self.graph_learner_init = graph_learn_func(in_dim, 
        #                             graph_learn_hidden_size,
        #                             topk=graph_learn_topk,
        #                             epsilon=graph_learn_epsilon,
        #                             num_pers=graph_learn_num_pers,
        #                             metric_type=self.graph_metric_type
        #                             )


        #     self.graph_learner = graph_learn_func(num_hidden,
        #                             graph_learn_hidden_size,
        #                             topk=graph_learn_topk,
        #                             epsilon=graph_learn_epsilon,
        #                             num_pers=graph_learn_num_pers,
        #                             metric_type=self.graph_metric_type
        #                             )
    
        # add by ssh
        self.guide_encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.updata_guide_model = 1
        self.decay = decay
        self.shadow = {}

        self._drop_edge_rate = 0.1
        self._add_rate  = 0.1 # cora 0.1比较好，这个比例有什么说法么，探索一下
        self._mask_rate = 0.4 # mask掉graph perb的比例 不能为0 要不nan
        self.undirected = True # 增加边是否无向增加
        self.use_gumbel_softmax = True
        self.mode = 'sim' # readd edges 的mode

        if concat_hidden:
            self.EdgeReEnhence = EdgeReEnhence(dec_in_dim * num_layers, dec_num_hidden, mode= self.mode)
        else:
            self.EdgeReEnhence = EdgeReEnhence(dec_in_dim, dec_num_hidden, mode= self.mode)

        self.StructDecoder = StructDecoder(dec_in_dim, dec_num_hidden)



    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == 'ce':
            criterion = ce_loss
        elif loss_fn == 'l2':
            criterion = l2_loss
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, graph_processed, adj_delete, x, add_rate, mask_rate=0.3,undirected=True):
        # mask node features 加了好像没啥用
        num_nodes = graph_processed.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            # 这里利用的BERT中的思想 BERT建议不总是用实际的[MASK]令牌替换“掩码”单词，而是用小概率(即15%或更小)保持不变或用另一个随机令牌替换它。
            # 对mask的节点一部分保持 一部分随机替换 一部分特征mask为0
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0


        out_x[token_nodes] +=  self.enc_mask_token

        # add perb edges
        graph_perbutaion = graph_processed.clone()

        if undirected:
            adj_delete = torch.triu(adj_delete).nonzero().T
        else:
            adj_delete = adj_delete.nonzero().T

            # random


        add_edge_idx = random_add_edges(adj_delete, add_rate)
        add_perb_edges_1,add_perb_edges_2 = adj_delete[0][add_edge_idx].to(x.device),adj_delete[1][add_edge_idx].to(x.device)
        graph_perbutaion.add_edges(add_perb_edges_1,add_perb_edges_2)
        # print(graph_perbutaion.adj().to_dense().max())  有2
        if undirected:
            graph_perbutaion = dgl.to_bidirected(graph_perbutaion.cpu()).to(x.device)
            add_perb_edges_src = torch.cat((add_perb_edges_1,add_perb_edges_2))
            add_perb_edges_dst = torch.cat((add_perb_edges_2,add_perb_edges_1))
        else:
            add_perb_edges_src = add_perb_edges_1
            add_perb_edges_dst = add_perb_edges_2

            # gumbel softmax 和random没啥区别

        # a1_all = self.struct_embed(graph_perbutaion,x)
        # a2_all = self.feat_embed(x)

        # a_1 = a1_all[adj_delete[0]]
        # a_2 = a2_all[adj_delete[0]]

        # a = torch.cat([a_1, a_2], dim=-1)
        # a = torch.relu(self.fc_out_1(a))

        # b_1 = a1_all[adj_delete[1]]
        # b_2 = a2_all[adj_delete[1]]

        # b = torch.cat([b_1, b_2], dim=-1)
        # b = torch.relu(self.fc_out_1(b))

        # edge_feat = torch.cat([a, b], dim=-1)
        # edge_feat = torch.relu(self.fc_out_2(edge_feat))
        # bernoulli_unnorm = self.fc_cat(edge_feat)
        # sampled_edge = gumbel_softmax(bernoulli_unnorm, temperature=0.5, hard=True)
        # sampled_edge = sampled_edge[..., 0]
        # sampled_edge_index = sampled_edge.nonzero().squeeze(-1)
        
        # num_sampled_edges = sampled_edge_index.shape[0]
        # perm = torch.randperm(num_sampled_edges, device=x.device)
        # num_add_edges = int(add_rate * num_sampled_edges)
        # add_edges_index = sampled_edge_index[list(perm[: num_add_edges])]
        
        # add_perb_edges_1 = torch.tensor(adj_delete[0])[list(add_edges_index)].to(x.device)
        # add_perb_edges_2 = torch.tensor(adj_delete[1])[list(add_edges_index)].to(x.device)
        
        # add_perb_edges_src = add_perb_edges_1
        # add_perb_edges_dst = add_perb_edges_2

        # graph_perbutaion.add_edges(add_perb_edges_1,add_perb_edges_2)
        # if undirected:
        #     graph_perbutaion = dgl.to_bidirected(graph_perbutaion.cpu()).to(x.device)

        #     add_perb_edges_src = torch.cat((add_perb_edges_1,add_perb_edges_2))
        #     add_perb_edges_dst = torch.cat((add_perb_edges_2,add_perb_edges_1))


        # 添加扰动边的操作结束 这里是不是可以设计一个loss根据XTX引导gumbel softmax的选择

        #设计方法从adj_delete中选出容易区分的边和不容易区分的边

        # 1.按照特征相似度，越低的越好区分
        # 2.用一个模型产出节点的特征，利用gumbul softmax 有些边只考虑特征可能相似度低，但是通过结构聚合之后相似度却比较高
        # 有些边只考虑特征可能相似度高，但是通过结构聚合之后相似度却比较低，通过一个合并这两个特征，得到一个权重
        # 一个用MLP 不考虑结构的特征 一个用GCN考虑纯结构

        return graph_perbutaion, out_x, mask_nodes ,(add_perb_edges_src,add_perb_edges_dst)

    def register(self):
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()


    def update(self):
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def guide_model_apply_shadow(self):
        for name, param in self.guide_encoder.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]


    def forward(self, graph_processed, adj_delete, x, epoch):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(graph_processed, adj_delete, x,epoch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, graph_processed, adj_delete ,x, epoch):

        # 原始矩阵cos相似度分析
        # x = x / torch.norm(x,dim=-1, keepdim=True)
        # z_cos = torch.mm(x,x.t())
        # print(z_cos.shape)
        # print(torch.sum(z_cos >= 0.0))
        ##########################

        graph_refine = graph_processed.clone()

        if self._drop_edge_rate > 0:
            graph_processed, masked_edges = drop_edge(graph_processed, self._drop_edge_rate, return_edges=True)
        else:
            graph_processed = graph_processed

        # 引导式增加扰动，希望一开始给一些不错的信息，但后面加大扰动的概率
        graph_preturbation, use_x, mask_nodes, (perb_edges_src,perb_edges_dst) = self.encoding_mask_noise(graph_processed, adj_delete, x ,self._add_rate, self._mask_rate,undirected = self.undirected)
        
        enc_rep, all_hidden = self.encoder(graph_preturbation, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # encoder矩阵cos相似度分析
        # enc_rep= enc_rep / torch.norm(enc_rep,dim=-1, keepdim=True)
        # enc_rep_cos = torch.mm(enc_rep,enc_rep.t())
        # print(enc_rep_cos.shape)
        # print(torch.sum(enc_rep_cos >= 0.8))

        # 根据同质性假设 是不是经过GCN之后更小了
        # print(self.feature_smoothing(graph_preturbation.adj().to_dense().cuda(),enc_rep))
        # 发现大部分cos相似度竟然都这么高！修改扰动边试试，扰动边比例越大，这个值越小（肯定，不小就麻烦了）
        # if epoch == 10:
        #     quit()
        
        ##########################


        # encoder出的embedding对graph_refine做一个增强
        # if perb_edges_src.numel() != 0 and epoch > 100: # 
        #     # 灵感 一直在如何在粗处理去掉的边上恢复出被误删的但有信息作用的边，可以再前期训练的时候先按照纯的graph refine固定，当模型具有一定的鲁棒能力之后，再开始进行edgereenhence
        #     # 但是如何衡量这个指标
        #     readd_edges_index = self.EdgeReEnhence(enc_rep, (perb_edges_src,perb_edges_dst), use_gumbel_softmax = self.use_gumbel_softmax)
        #     graph_refine.add_edges(perb_edges_src[readd_edges_index],perb_edges_dst[readd_edges_index])
        # ####

        if epoch % self.updata_guide_model == 0:
            self.guide_model_apply_shadow()
        
        ############    graph learning ...
        # if self.graph_learn and self.graph_learn_regularization:
        #     print("In epoch: %s, graph learning... ")
        #     loss1 = 0 
        #     init_node_vec = x

        #     cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner_init, x, x.device)

        #     guide_enc_rep, guide_all_hidden = self.guide_encoder(cur_adj,init_node_vec)

        #     if self._concat_hidden:
        #         guide_enc_rep = torch.cat(guide_all_hidden, dim=1)

        #     guide_enc_rep = guide_enc_rep.detach()

        #     loss1 += self.add_graph_loss(cur_raw_adj, init_node_vec)

        #     first_raw_adj, first_adj = cur_raw_adj, cur_adj

        #     pre_raw_adj = cur_raw_adj
        #     pre_adj = cur_adj

        #     loss = 0
        #     iter_ = 0

        #     while self.graph_learn and (iter_ == 0 or self.diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > self.eps_adj) and iter_ < self.max_iter:
        #         iter_ += 1
        #         pre_adj = cur_adj
        #         pre_raw_adj = cur_raw_adj

        #         cur_raw_adj, cur_adj = self.learn_graph(self.graph_learner, guide_enc_rep, guide_enc_rep.device())

        #         guide_enc_rep, guide_all_hidden = self.guide_encoder(cur_adj,init_node_vec)
            
        #         if self._concat_hidden:
        #             guide_enc_rep = torch.cat(guide_all_hidden, dim=1)

        #         guide_enc_rep = guide_enc_rep.detach()

        #         loss += self.add_graph_loss(cur_raw_adj, init_node_vec)
        #         loss += self.SquaredFrobeniusNorm(cur_raw_adj - pre_raw_adj) * self.graph_learn_ratio


        #     if iter_ > 0:
        #         loss_graph = loss / iter_ + loss1
        #     else:
        #         loss_graph = loss1
        ##############
        # else:
        guide_enc_rep, guide_all_hidden = self.guide_encoder(graph_refine, x, return_hidden=True)
        if self._concat_hidden:
            guide_enc_rep = torch.cat(guide_all_hidden, dim=1)

        guide_enc_rep = guide_enc_rep.detach()


        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        
        recon_struct = self.StructDecoder(rep, sigmoid=True)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(graph_preturbation, rep)

        # print(self.criterion(recon[mask_nodes], guide_enc_rep[mask_nodes]))
        # print(kl_categorical(graph_refine.adj().to_dense().cuda()[mask_nodes],recon_struct[mask_nodes]))

        loss =  self.criterion(recon[mask_nodes], guide_enc_rep[mask_nodes]) + 100 *  kl_categorical(graph_refine.adj().to_dense().cuda(),recon_struct)   #+  loss_graph #+ 2 * ce_loss(mask_out,perb_out)

        return loss



    # coding ...
    def learn_graph(self, graph_learner, node_features, device = None, graph_skip_conn=None, graph_include_self=False, init_adj=None):
        raw_adj = graph_learner(node_features, device=device)

        if self.graph_metric_type in ('kernel', 'weighted_cosine'):
            assert raw_adj.min().item() >= 0
            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

        elif self.graph_metric_type == 'cosine':
            adj = (raw_adj > 0).float()
            adj = normalize_adj(adj)

        else:
            adj = torch.softmax(raw_adj, dim=-1)

        if graph_skip_conn in (0, None):
            if graph_include_self:
                adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
        else:
            adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

        return raw_adj, adj



    def add_graph_loss(self, out_adj, features):
        # Graph regularization
        graph_loss = 0
        L = torch.diagflat(torch.sum(out_adj, -1)) - out_adj
        graph_loss += self.smoothness_ratio * torch.trace(torch.mm(features.transpose(-1, -2), torch.mm(L, features))) / int(np.prod(out_adj.shape))
        ones_vec = to_cuda(torch.ones(out_adj.size(-1)), self.device)
        graph_loss += -self.degree_ratio * torch.mm(ones_vec.unsqueeze(0), torch.log(torch.mm(out_adj, ones_vec.unsqueeze(-1)) + VERY_SMALL_NUMBER)).squeeze() / out_adj.shape[-1]
        graph_loss += self.sparsity_ratio * torch.sum(torch.pow(out_adj, 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def SquaredFrobeniusNorm(self,X):
        return torch.sum(torch.pow(X, 2)) / int(np.prod(X.shape))

    def diff(self,X, Y, Z):
        assert X.shape == Y.shape
        diff_ = torch.sum(torch.pow(X - Y, 2))
        norm_ = torch.sum(torch.pow(Z, 2))
        diff_ = diff_ / torch.clamp(norm_, min=VERY_SMALL_NUMBER)
        return diff_


    # use for analysis
    def feature_smoothing(self, adj, X):
        adj = (adj.t() + adj)/2
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        # L = r_mat_inv @ L
        L = r_mat_inv @ L @ r_mat_inv

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat



    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])