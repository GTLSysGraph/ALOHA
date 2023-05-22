import torch
import torch.nn as nn
from typing import Optional
from itertools import chain
from functools import partial
from ..utils import create_norm, drop_edge 
from .loss_func import sce_loss
from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .graphformer_decoder import * 
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..diffusion.position_encoding import PositionalEncoding
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
            num_nodes:int,
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
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

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





        self.use_graphformer_decoder = False
        
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        # remask 可见patch的emedding
        self.re_enc_mask_token = nn.Parameter(torch.FloatTensor(size=(1, in_dim)))
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.re_enc_mask_token, gain=gain)


        # 如果decoder用GNN，则需要用下面的语句 同时self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, in_dim, bias=False) ，把out dim的dec_in_dim改成in_dim
        if self.use_graphformer_decoder == True:
            # 如果用graph former，cross attention不需要encoder的特征和noise的维度一样
            dec_in_dim_final = dec_in_dim
        else:
            # 如果用GNN，需要保持两部分的维度一致，所以不同于graphmae，因为在这里要把加噪声的特征和encoder输出的可见特征拼接再一起，所以encoder部分要需要重新映射为in_dim
            dec_in_dim_final = in_dim
        #    

        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim_final, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim_final, bias=False)

        # self.noise_linear = nn.Linear(in_dim, dec_in_dim_final, bias=False)

        # build decoder for attribute prediction
        # 如果把decoder的model用作扩散模型的话，输入和输出都是in_dim
        # 但是decoder的有一部分是可见patch的在encoder后的embedding进行mlp转换回来的，为了辅助decoder训练

        # 注意，在decoder中用的是一层，并且是nhead_out 只有1个头，做实验发现如果nhead_out使用多头，并且concat_out为求平均的话效果会很差，encoder基本没有学习到什么
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim_final,
            num_hidden=dec_num_hidden,
            out_dim=in_dim, # in_dim 
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





        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        #### add by ssh
        self.predict_noises = False
        self.timestep = 10000
        self.beta_schedule = 'linear'
        self.gaussian_diffusion = GaussianDiffusion(self.timestep, self.beta_schedule)
        self.position_encoding =  PositionalEncoding(in_dim)

        ## graphformer realize
        self.graphformer_hidden = 512
        self.graphformer_heads  = 1 
        self.last_state_only = True
        self.num_decoder_layers = 1
        # self.cross_attention_input_dim = []
        self.cross_attention_input_dim = [self.graphformer_hidden, dec_in_dim, dec_in_dim]
        self.graphformer_decoder = GraphDecoder(
                                                in_dim,
                                                self.graphformer_hidden,
                                                self.graphformer_heads,
                                                dec_in_dim_final,
                                                self.last_state_only,
                                                self.cross_attention_input_dim,
                                                self.num_decoder_layers
                                            )



    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, x, n_steps,mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        # 无差别remask node
        # remask_rate = 0.5
        # perm_remask = torch.randperm(num_nodes, device=x.device)
        # num_remask_nodes = int(remask_rate * num_nodes)
        # remask_nodes = perm_remask[: num_remask_nodes]


        # 只从mask里面挑一部分remask
        remask_rate = 0.5
        perm_index = torch.randperm(num_mask_nodes, device=x.device)
        num_remask_nodes = int(remask_rate * num_mask_nodes)
        remask_nodes = mask_nodes[perm_index[: num_remask_nodes]]


        if self._replace_rate > 0:
            # 这里利用的BERT中的思想 BERT建议不总是用实际的[MASK]令牌替换“掩码”单词，而是用小概率(即15%或更小)保持不变或用另一个随机令牌替换它。
            # 对mask的节点一部分保持 一部分随机替换 一部分特征mask为0
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_encoder_x = x.clone()
            out_encoder_x[token_nodes] = 0.0
            out_encoder_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_encoder_x = x.clone()
            token_nodes = mask_nodes
            out_encoder_x[mask_nodes] = 0.0
        #


        out_diffusion_x = x.clone()
        # add noise
        timestep_size = out_diffusion_x[mask_nodes].shape[0]
        # 注意初始值不为0，会做分母
        mask_nodes_t = torch.randint(8000, n_steps, size=(timestep_size,)).long().to(out_diffusion_x.device)
        
        out_diffusion_x[mask_nodes] = self.gaussian_diffusion.q_sample(out_diffusion_x[mask_nodes],mask_nodes_t)

        mask_positon_encoding = self.position_encoding(mask_nodes_t)
        mask_positon_encoding = mask_positon_encoding.to(out_diffusion_x.device)
        out_diffusion_x[mask_nodes] += mask_positon_encoding

        out_diffusion_x[mask_nodes] +=  self.enc_mask_token

        use_g = g.clone()

        return use_g, perm, out_encoder_x, out_diffusion_x, mask_nodes_t, (mask_nodes,remask_nodes,keep_nodes)

    def forward(self, g, x, epoch):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x, epoch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, g, x, epoch):
        pre_use_g, perm, out_encoder_x, out_diffusion_x, mask_nodes_t, (mask_nodes, remask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self.timestep,self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, out_encoder_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # 这里encoder得到的是没有被noise的可见embedding
        # ---- attribute reconstruction ----

        # 如果用G former ,cross attention，两部分维度可以不一样
        rep = self.encoder_to_decoder(enc_rep)
        # 不使用mlp，进行补0扩充
        # rep = torch.cat((rep,torch.zeros(rep.shape[0],out_diffusion_x.shape[-1]-rep.shape[-1]).to(out_diffusion_x.device)),dim = -1)

        if self.use_graphformer_decoder == True:
            noise_feature =   out_diffusion_x[mask_nodes]  #* (1 /mask_nodes_t.unsqueeze(-1)) + self.enc_mask_token 
            # self_whole_mask = pre_use_g.adj().to_dense().to(rep.device)
            self_mask = pre_use_g.adj().to_dense()[mask_nodes,:][:,mask_nodes].to(rep.device)
            cross_mask = pre_use_g.adj().to_dense()[mask_nodes,:][:,keep_nodes].to(rep.device)
            recon = self.graphformer_decoder(noise_feature, rep[keep_nodes], self_mask, cross_mask) # graph, noise_feature, resist_enc_embed
        
            x_rec = recon
    
        else:   
            # out_diffusion_x  = self.noise_linear(out_diffusion_x)
            # self_mask = pre_use_g.adj().to_dense()[mask_nodes,:][:,mask_nodes].to(rep.device)
            # cross_mask = pre_use_g.adj().to_dense()[mask_nodes,:][:,keep_nodes].to(rep.device)
            # rep_dec = self.graphformer_decoder(out_diffusion_x[mask_nodes], rep[keep_nodes], self_mask, cross_mask)
            # rep = torch.zeros(x.shape[0],rep_dec.shape[1]).to(rep.device)
            # rep[mask_nodes] = rep_dec

            rep[keep_nodes] = rep[keep_nodes]  # 给一定的缩放比例貌似能提升效果，更好的关注生成部分 0.0的时候不能重建，这就证明需要encoder提供的信息，但是直接用效果又不太好，得缩放简单提供
            rep[mask_nodes] =  out_diffusion_x[mask_nodes] * (1 /mask_nodes_t.unsqueeze(-1)) + self.enc_mask_token  # 这里重新用enc_mask_token比再重新用一个re_enc_mask_token要好很多
            #  
            # # ######### 看一下是不是mask不为0就会影响结果
            if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
                rep[remask_nodes] = 0
            ###############################################

            # rep_use_for_against = rep.clone()
            # rep_use_for_against[keep_nodes] == 0

            if self._decoder_type in ("mlp", "liear") :
                recon = self.decoder(rep)
            else:
                recon = self.decoder(pre_use_g, rep)
                rep[keep_nodes] = 0
                recon_against = self.decoder(pre_use_g, rep)

            x_rec =  recon[mask_nodes]
        x_init = x[mask_nodes]
        

        # x_against = recon_against[mask_nodes]
        # # 希望decoder输出的keep node特征之间具有很好的同质性
        # keep_nodes_adj = pre_use_g.adj().to_dense()[mask_nodes,:][:,mask_nodes].to(rep.device)


        # 如果用gat做扩散模型感觉太简单了，这里试一下模仿transformer或者Unet
        if self.predict_noises:
            # generate random noise
            noise = torch.randn_like(x_init)
            # get x_t
            loss = self.criterion(noise, x_rec)
        else:

            loss = self.criterion(x_rec, x_init)  +  0.1 * self.criterion(recon_against[mask_nodes], -5 * x[perm][mask_nodes])
        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

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


    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])