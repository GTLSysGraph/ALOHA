import os.path as osp
from torch_geometric.nn import GCNConv

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_edge
import torch_geometric.transforms as T

from datasets_pyg.data_pyg import get_dataset
from model_zoo.GCL.GRACE.GRACE import *
from time import perf_counter as t
from model_zoo.GCL.GRACE.eval import label_classification
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops

import copy
from utils import *
from tqdm import tqdm
from model_zoo.GCL.GRACE.build_easydict import *


def train_inductive(model,optimizer,drop_edge_rate_1,drop_edge_rate_2,drop_feature_rate_1,drop_feature_rate_2,epoch,train_loader):
    model.train()
    optimizer.zero_grad()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = 0
    for batch in train_loader:
        edge_index_1 = dropout_edge(batch.edge_index, p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_edge(batch.edge_index, p=drop_edge_rate_2)[0]
        x_1 = drop_feature(batch.x, drop_feature_rate_1)
        x_2 = drop_feature(batch.x, drop_feature_rate_2)
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.loss(z1, z2, batch_size = batch.batch_size)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples


def train_transductive(model:Model,optimizer, drop_edge_rate_1,drop_edge_rate_2,drop_feature_rate_1,drop_feature_rate_2,x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test_inductive(model, subgraph_loader, val_mask, test_mask, y, device):
    model.eval()
    pbar = tqdm(total=int(len(subgraph_loader.dataset)))
    pbar.set_description('Evaluating')

    accs = []
    ys = []
    for batch in subgraph_loader:
        z = model(batch.x.to(device), batch.edge_index.to(device))
        ys.append(z[:batch.batch_size])
        pbar.update(batch.batch_size)
    pbar.close()
    ys = torch.cat(ys, dim=0)
    ys = ys.argmax(dim=-1)

    for mask in [val_mask, test_mask]:
        accs.append(int((ys[mask] == y[mask]).sum()) / int(mask.sum()))
    print('val_acc : {}, test_acc: {}'.format(accs[0],accs[1]))
    # label_classification(ys, y, ratio=0.1, shuffle = False)
    


def test_transductive(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)
    label_classification(z, y, ratio=0.1, shuffle = True)



def Train_GRACE_nodecls(margs):
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"

    dataset_name = margs.dataset
    path = osp.expanduser('/home/songsh/GCL/datasets_pyg')
    if dataset_name.split('-')[0] == 'Attack':
        attackmethod = margs.attack.split('-')[0]
        attackptb    = margs.attack.split('-')[1]
        path = osp.expanduser('/home/songsh/GCL/datasets_pyg/Attack_data')
        dataset = get_dataset(path, dataset_name, attackmethod, attackptb)
    else:
        path = osp.join(path, dataset_name)
        dataset = get_dataset(path, dataset_name)

    data = dataset[0]
    data = data.to(device)


    inductive_task = (margs.mode == 'inductive')

    # transform = T.Compose([
    #     T.ToUndirected(),
    #     T.ToDevice(device),
    # ])

    # data = transform(dataset[0])

    MDT = build_easydict()
    config     = MDT['MODEL']['PARAM']
    if config.use_cfg:
        config = load_best_configs(config, dataset_name, "/home/songsh/GCL/model_zoo/GCL/GRACE/config.yaml")

  
    torch.manual_seed(config['seed'])
    
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']



    # https://www.cnblogs.com/mercurysun/p/16869877.html 参考graphsage的代码自己改编的inductive
    if inductive_task:
        batch_size = config['batch_size']
        # 传入mask或者id到loader都可以 这里要注意，dataloader是从cpu取数据放到gpu上的，不能提前放到cuda上，否则会段错误
        if dataset_name == 'ogbn-arxiv':
            split_idx = dataset.get_idx_split()
            train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            train_mask = index_to_mask(train_idx, data.num_nodes)
            val_mask = index_to_mask(val_idx, data.num_nodes)
            test_mask = index_to_mask(test_idx, data.num_nodes)
        else:
            train_mask, val_mask, test_mask = data.train_mask, data.val_mask ,data.test_mask
            train_mask = train_mask.cpu()
            val_mask   = val_mask.cpu()
            test_mask  = test_mask.cpu()

        # train_mask, test_mask, val_mask = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

        # split_idx = dataset.get_idx_split()      
        # train_idx = split_idx['train']
        # valid_idx = split_idx['valid']
        # test_idx = split_idx['test']

        # 如果用小图做inductive任务 肯定比用transductive差 因为训练集的数据太小 其他点不可见，但transductive中全部节点可见
        kwargs = {'batch_size': batch_size}
        train_loader = NeighborLoader(data, input_nodes=train_mask,
                                    num_neighbors=[25,10], shuffle=True, **kwargs)
        subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                    num_neighbors=[-1], shuffle=False, **kwargs)
        

    # model
    encoder =   Encoder(dataset.num_features, num_hidden, activation,base_model=base_model, k=num_layers).to(device)
    model   =   Model(encoder, num_hidden, num_proj_hidden, tau).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # train
    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        if inductive_task:
            loss = train_inductive(model,optimizer,drop_edge_rate_1,drop_edge_rate_2,drop_feature_rate_1,drop_feature_rate_2,epoch,train_loader)
        else:
            loss = train_transductive(model,optimizer, drop_edge_rate_1,drop_edge_rate_2,drop_feature_rate_1,drop_feature_rate_2, data.x, data.edge_index)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    # test
    print("=== Final ===")
    if inductive_task:
        test_inductive(model, subgraph_loader, val_mask, test_mask, data.y,device)
    else:
        test_transductive(model, data.x, data.edge_index, data.y, final=True)


