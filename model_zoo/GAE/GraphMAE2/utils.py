import os
import argparse
import random
import psutil
import yaml
import logging
from functools import partial
from tensorboardX import SummaryWriter
import wandb

import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim


import dgl
import dgl.function as fn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x
    return func

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        # print("Identity norm")
        return None


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
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer


def show_occupied_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    graph = graph.remove_self_loop()

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src, dst = graph.edges()

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng


def visualize(x, y, method="tsne"):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
        
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    
    if method == "tsne":
        func = TSNE(n_components=2)
    else:
        func = PCA(n_components=2)
    out = func.fit_transform(x)
    plt.scatter(out[:, 0], out[:, 1], c=y)
    plt.savefig("vis.png")
    

def load_best_configs(args, dataset_name):
    config_path = os.path.join("./model_zoo/GAE/GraphMAE2/configs", f"{dataset_name}.yaml")
    with open(config_path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    logging.info(f"----- Using best configs from {config_path} -----")

    return args



def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    scheduler = np.concatenate((warmup_schedule, schedule))
    assert len(scheduler) == epochs * niter_per_ep
    return scheduler

    

# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
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


class WandbLogger(object):
    def __init__(self, log_path, project, args):
        self.log_path = log_path
        self.project = project
        self.args = args
        self.last_step = 0
        self.project = project
        self.start()

    def start(self):
        self.run = wandb.init(config=self.args, project=self.project)

    def log(self, metrics, step=None):
        if not hasattr(self, "run"):
            self.start()
        if step is None:
            step = self.last_step
        self.run.log(metrics)
        self.last_step = step

    def finish(self):
        self.run.finish()