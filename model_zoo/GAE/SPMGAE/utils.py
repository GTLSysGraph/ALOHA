import random
import torch
import numpy as np
from torch import optim as optim
import os
import logging
import torch.nn as nn
from tensorboardX import SummaryWriter
from functools import partial
import dgl
import yaml
from collections import namedtuple, Counter
from datasets_dgl.data_dgl import load_graph_data
import torch.nn.functional as F


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def compute_edge_sim(edges,features,sim_mode='cosine'):
    if sim_mode == 'jaccard':
        intersection = torch.count_nonzero(features[edges[0]] * features[edges[1]],dim = 1)
        edges_sim = intersection * 1.0 / (torch.count_nonzero(features[edges[0]],dim=1) + torch.count_nonzero(features[edges[1]],dim=1) - intersection)
    elif sim_mode == 'cosine':
        edges_sim = F.cosine_similarity(features[edges[0]], features[edges[1]], dim=1)
        edges_sim = 0.5 + 0.5 * edges_sim # # 取值范围(-1 ~ 1),0.5 + 0.5 * result 归一化(0 ~ 1)
    elif sim_mode == 'pearson':
        avg0 = torch.mean(features[edges[0]],dim=1).unsqueeze(-1)
        avg1 = torch.mean(features[edges[1]],dim=1).unsqueeze(-1)
        edges_sim = F.cosine_similarity(features[edges[0]] - avg0 , features[edges[1]] - avg1, dim=1)
        edges_sim = 0.5 + 0.5 * edges_sim # # 皮尔逊相关系数的取值范围(-1 ~ 1),0.5 + 0.5 * result 归一化(0 ~ 1)
    
    print(max(edges_sim))
    print(min(edges_sim))
    return edges_sim



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer

class TBLogger(object):
    def __init__(self, log_path="./model_zoo/GAE/GraphMAE/logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity

def add_ptb_edges(graph, add_edges, add_prob, undirected):
    E = len(add_edges[0])
    add_rates = torch.FloatTensor(np.ones(E) * add_prob)
    add_masks = torch.bernoulli(add_rates).bool()
    add_masks_idx = add_masks.nonzero().squeeze(1)

    add_edges_src = add_edges[0][add_masks_idx]
    add_edges_dst = add_edges[1][add_masks_idx]

    graph.add_edges(add_edges_src,add_edges_dst)
    if undirected == True:
        graph = dgl.to_bidirected(graph.cpu()).to('cuda')
    return graph, (add_edges_src,add_edges_dst)




def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    graph = graph.remove_self_loop() # 不删除自环边，只drop非自环边
    keep_mask_idx, masked_mask_idx = mask_edge(graph, drop_rate)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[keep_mask_idx]
    ndst = dst[keep_mask_idx]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[masked_mask_idx]
    ddst = dst[masked_mask_idx]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng

def mask_edge(graph, mask_prob):
    E = graph.num_edges()
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    keep_masks = torch.bernoulli(1 - mask_rates).bool()
    keep_mask_idx = keep_masks.nonzero().squeeze(1)
    masked_masks = ~keep_masks
    masked_mask_idx = masked_masks.nonzero().squeeze(1)
    return keep_mask_idx, masked_mask_idx

def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def load_best_configs(param, dataset_name, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if dataset_name not in configs:
        logging.info("Best args not found")
        return param

    logging.info("Using best configs")
    configs = configs[dataset_name]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(param, k, v)
    print("------ Use best configs ------")
    return param



class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias





def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = load_graph_data(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())
            
            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                
                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])
    
    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)