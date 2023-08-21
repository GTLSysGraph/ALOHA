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
from .edgedecoder import EdgeDecoder
from ..utils import add_ptb_edges
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

# from model_zoo.GAE.MaskGAE.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
# encoder = GNNEncoder(1433, 512, 512,
#                     num_layers=2, dropout=0.8,
#                     bn=True, layer='gcn', activation="elu").to('cuda')



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
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            #
            edge_hidden        = 64,
            updata_guide_model = 10,
            decay              = 0.95,
            undirected         = False,
            add_rate           = 0.4,
            gamma              = 10,
            beta               = 0.001,
            type_graph4recon   = "refine"
            #
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

        ########### add by ssh
        self._undirected        = undirected
        self._add_rate          = add_rate
        self.edge_hidden        = edge_hidden
        self.updata_guide_model = updata_guide_model
        self.decay              = decay
        self.gamma              = gamma
        self.beta               = beta
        self.type_graph4recon   = type_graph4recon
        self.shadow = {}
        ###########

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


        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
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

        if self._concat_hidden == True:
            self.edgedecoder = EdgeDecoder(dec_in_dim * num_layers, edge_hidden)
        else:
            self.edgedecoder = EdgeDecoder(dec_in_dim, edge_hidden)


        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.register()


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
    
    def encoding_mask_noise(self, use_g, del_edges, x, add_prob, mask_rate, undirected):
        num_nodes = use_g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
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
            # 注意如果为0 ，noise_nodes里面是-0,和0一样，就包含了所有的mask_modes，格式不匹配
            if num_noise_nodes != 0:
                out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        # 这里为啥不扩展啊 这里不加效果确实不如加了好，为啥啊，感觉没道理啊
        out_x[token_nodes] +=  self.enc_mask_token

        # add perb edges
        use_g, (add_edges_src,add_edges_dst) = add_ptb_edges(use_g, del_edges, add_prob, undirected)

        return use_g, out_x, (mask_nodes, keep_nodes), (add_edges_src,add_edges_dst)


    def register(self):
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()


    def update(self):
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.cpu() + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def guide_model_apply_shadow(self):
        for name, param in self.guide_encoder.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]


    def forward(self, graph_refine, del_edges, x, epoch):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(graph_refine, del_edges, x, epoch)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def random_negative_sampler(self,edge_index, num_nodes, num_neg_samples):
        neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
        return neg_edges


    def ce_loss(self, pos_out, neg_out):
        pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
        return pos_loss + neg_loss

    
    def mask_attr_prediction(self, graph_refine, del_edges, x, epoch):
        use_g = graph_refine.clone()

        if self._drop_edge_rate > 0:
            use_g,  masked_edges = drop_edge(use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = use_g

        graph_preturbation, use_x, (mask_nodes, keep_nodes), (perb_edges_src,perb_edges_dst) = self.encoding_mask_noise(use_g, del_edges, x, self._add_rate, self._mask_rate, undirected = self._undirected)

        enc_rep, all_hidden = self.encoder(graph_preturbation, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)


        if epoch % self.updata_guide_model == 0:
            self.update()
            self.guide_model_apply_shadow()

        self.guide_encoder = self.guide_encoder.to('cuda')
        guide_enc_rep, guide_all_hidden = self.guide_encoder(graph_refine, x, return_hidden=True)
        if self._concat_hidden:
            guide_enc_rep = torch.cat(guide_all_hidden, dim=1)
        guide_enc_rep = guide_enc_rep.detach()
        

        # **************************** attribute reconstruction ***************************
        rep = self.encoder_to_decoder(enc_rep)
    

        # **************************** sturct reconstruction ***************************
        pos_edges        =  torch.stack((graph_refine.edges()[0],graph_refine.edges()[1]),dim=0)
        neg_edges        =  self.random_negative_sampler(pos_edges, num_nodes=graph_refine.num_nodes(), num_neg_samples =pos_edges.view(2, -1).size(1),).view_as(pos_edges) 
        
        pos_edge_out  = self.edgedecoder(enc_rep,  pos_edges,     sigmoid=False )
        neg_edges_out = self.edgedecoder(enc_rep,  neg_edges,     sigmoid=False )
        loss_struct   = self.ce_loss(pos_edge_out, neg_edges_out)

        # **************************** recon features **************************************
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        # 加一个判断，graph4recon = graph_preturbation if not attack else graph_refine
        if self.type_graph4recon == "refine":
            graph4recon = graph_refine
        else:
            graph4recon = graph_preturbation


        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(graph4recon, rep) # 扰动小的时候用graph_preturbation 用pertuerbation比refine要好，猜测原因，在cora的时候貌似更喜欢对扰动矩阵重建，扰动小的时候其实所有的边都是有用的，所以其实graph perb比refine多了更多的有用信息来帮助重建，因为加的边其实并不是扰动边


        x_init = x[mask_nodes]
        x_rec  = recon[mask_nodes]


    
        # mask ? keep ? all?
        loss =   self.criterion(x_rec, x_init)  +  self.gamma * F.mse_loss(enc_rep[mask_nodes], guide_enc_rep[mask_nodes]) + self.beta * loss_struct
        # loss = loss_struct
        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])