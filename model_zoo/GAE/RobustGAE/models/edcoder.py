import torch
import torch.nn as nn
from typing import Optional
from itertools import chain
from functools import partial
from ..utils import create_norm, drop_edge, gumbel_softmax
from .loss_func import sce_loss,ce_loss
from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
import dgl
from .auxiliary_model import EdgeDecoder
from ..utils import random_add_edges

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
            updata_guide_model = 10,
            decay = 0.6,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(PreModel, self).__init__()

        self._add_rate  = 0.1
        self._mask_rate = mask_rate

        ###
        self._struct_type = 'gcn'
        struct_num_hidden = 128
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

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            # 本身的目的是重建特征 out dim 边embedding？试一下 或者重新定义个edgecoder
            out_dim=512,
            num_layers=1,
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


        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)


        ####################################################### add by ssh
        # MLP for origin features
        self.updata_guide_model = updata_guide_model
        self.decay = decay
        self.shadow = {}
        

        self.fc_cat = nn.Linear(struct_num_hidden, 2)
        self.fc_out_1 = nn.Linear(struct_num_hidden * 2, struct_num_hidden)
        self.fc_out_2 = nn.Linear(struct_num_hidden * 2, struct_num_hidden)

        self.feat_embed   = nn.Linear(in_dim, struct_num_hidden)
        self.struct_embed =   setup_module(
            m_type=self._struct_type,
            enc_dec="struct",
            in_dim=in_dim,
            num_hidden=struct_num_hidden,
            out_dim=struct_num_hidden,
            num_layers=num_layers,
            nhead=1,
            nhead_out=1,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # self.EdgeDecoder = EdgeDecoder(dec_in_dim, dec_num_hidden)
    


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
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, graph_processed, adj_delete, x, add_rate, undirected=True):

        graph_perbutaion = graph_processed.clone()

        if undirected:
            adj_delete = torch.triu(adj_delete).nonzero().T
        else:
            adj_delete = adj_delete.nonzero().T

        # random

        add_edge_idx = random_add_edges(adj_delete, add_rate)
        add_perb_edges_1,add_perb_edges_2 = adj_delete[0][add_edge_idx].to(x.device),adj_delete[1][add_edge_idx].to(x.device)

        graph_perbutaion.add_edges(add_perb_edges_1,add_perb_edges_2)
        if undirected:
            graph_perbutaion = dgl.to_bidirected(graph_perbutaion.cpu()).to(x.device)

            add_perb_edges_src = torch.cat((add_perb_edges_1,add_perb_edges_2))
            add_perb_edges_dst = torch.cat((add_perb_edges_2,add_perb_edges_1))
        

        # gumbel softmax

        # a1_all = self.struct_embed(graph_refine,x)
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

        # graph_refine.add_edges(add_perb_edges_1,add_perb_edges_2)
        # if undirected:
        #     graph_refine = dgl.to_bidirected(graph_refine.cpu()).to(x.device)

        #     add_perb_edges_src = torch.cat((add_perb_edges_1,add_perb_edges_2))
        #     add_perb_edges_dst = torch.cat((add_perb_edges_2,add_perb_edges_1))


        # 添加扰动边的操作结束 这里是不是可以设计一个loss根据XTX引导gumbel softmax的选择

        #设计方法从adj_delete中选出容易区分的边和不容易区分的边

        # 1.按照特征相似度，越低的越好区分
        # 2.用一个模型产出节点的特征，利用gumbul softmax 有些边只考虑特征可能相似度低，但是通过结构聚合之后相似度却比较高
        # 有些边只考虑特征可能相似度高，但是通过结构聚合之后相似度却比较低，通过一个合并这两个特征，得到一个权重
        # 一个用MLP 不考虑结构的特征 一个用GCN考虑纯结构

        return graph_perbutaion, x, (add_perb_edges_src,add_perb_edges_dst)

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

        graph_refine = graph_processed.clone()

        if self._drop_edge_rate > 0:
            graph_processed, masked_edges = drop_edge(graph_processed, self._drop_edge_rate, return_edges=True)
        else:
            graph_processed = graph_processed

        # 引导式增加扰动，希望一开始给一些不错的信息，但后面加大扰动的概率
        graph_preturbation, use_x, (perb_edges_src,perb_edges_dst) = self.encoding_mask_noise(graph_processed, adj_delete, x ,self._add_rate, undirected = True)

        if epoch % self.updata_guide_model == 0:
            self.guide_model_apply_shadow()

        guide_enc_rep, guide_all_hidden = self.guide_encoder(graph_refine, use_x, return_hidden=True)
        if self._concat_hidden:
            guide_enc_rep = torch.cat(guide_all_hidden, dim=1)

        guide_enc_rep = guide_enc_rep.detach()

        enc_rep, all_hidden = self.encoder(graph_preturbation, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        # perb_edges = (perb_edges_src,perb_edges_dst)

        # perb_out = self.EdgeDecoder(rep, perb_edges, sigmoid=False)

        # masked_out = self.EdgeDecoder(rep, masked_edges, sigmoid=False)

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(graph_preturbation, rep)

        # masked_out = self.EdgeDecoder(recon, masked_edges, sigmoid=False)

        loss = self.criterion(recon, guide_enc_rep)

        return loss
        # if self._decoder_type not in ("mlp", "linear"):
        #     # * remask, re-mask
        #     rep[mask_nodes] = 0

        # if self._decoder_type in ("mlp", "liear") :
        #     recon = self.decoder(rep)
        # else:
        #     recon = self.decoder(pre_use_g, rep)

        # x_init = x[mask_nodes]
        # x_rec = recon[mask_nodes]

        # loss = self.criterion(x_rec, x_init)
        # return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])