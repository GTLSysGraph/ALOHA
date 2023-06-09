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
from .eval import label_classification,linear_probe_nodeclas
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import add_self_loops

import copy
from utils import *
from tqdm import tqdm
from model_zoo.GCL.GRACE.build_easydict import *


def train_inductive(model,optimizer,drop_edge_rate_1,drop_edge_rate_2,drop_feature_rate_1,drop_feature_rate_2,epoch,train_loader):
    # model.train()
    # optimizer.zero_grad()

    # pbar = tqdm(total=int(len(train_loader.dataset)))
    # pbar.set_description(f'Epoch {epoch:02d}')

    # total_loss = total_examples = 0
    # for batch in train_loader:
    #     edge_index_1 = dropout_edge(batch.edge_index, p=drop_edge_rate_1)[0]
    #     edge_index_2 = dropout_edge(batch.edge_index, p=drop_edge_rate_2)[0]
    #     x_1 = drop_feature(batch.x, drop_feature_rate_1)
    #     x_2 = drop_feature(batch.x, drop_feature_rate_2)
    #     z1 = model(x_1, edge_index_1)
    #     z2 = model(x_2, edge_index_2)
    #     loss = model.loss(z1, z2, batch_size = batch.batch_size)
    #     loss.backward()
    #     optimizer.step()

    #     total_loss += float(loss) * batch.batch_size
    #     total_examples += batch.batch_size
    #     pbar.update(batch.batch_size)
    # pbar.close()

    # return total_loss / total_examples
    print('wait realise....')

def train_transductive(model:Model,optimizer, drop_edge_rate_1,drop_edge_rate_2,drop_feature_rate_1,drop_feature_rate_2,x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=128)
    loss.backward()
    optimizer.step()

    return loss.item()



# 有问题的代码，需要重新实现，摆脱label_classification
def test_inductive(model, val_loader, test_loader, val_mask, test_mask, y, device):
    # model.eval()
    # # val 太大了，cat会内存不足
    # for loader in [val_loader, test_loader]:
    #     pbar = tqdm(total=int(len(loader.dataset)))
    #     pbar.set_description('Evaluating val' if loader == val_loader else 'Evaluting test')
    #     ys = []
    #     nid = []
    #     F1Mi = []
    #     F1Ma =[]
    #     num_record = 0
    #     for batch in loader:
    #         z = model(batch.x.to(device), batch.edge_index.to(device))
    #         ys.append(z[:batch.batch_size])
    #         nid.append(batch.input_id.unsqueeze(-1))
    #         num_record += z.shape[0]

    #         if num_record > 4000:
    #             ys  = torch.cat(ys, dim=0)
    #             nid = torch.cat(nid, dim=0)
    #             # 这样是不行的，因为只有部分的标签，按时在label classfication里要onehot，必须要全部的标签才可以
    #             res_mid = label_classification_no_repeat(ys, y[nid], ratio=0.1, shuffle = False)
    #             F1Mi.append(res_mid['F1Mi'])
    #             F1Ma.append(res_mid['F1Ma'])
    #             ys  = []
    #             nid = []
    #             num_record = 0

    #         pbar.update(batch.batch_size)
    #     pbar.close()

    #     final_f1mi = np.mean(F1Mi)
    #     final_f1ma = np.mean(F1Ma)

    #     if loader == val_loader:
    #         print('val F1Mi : {}, val F1Ma : {}'.format(final_f1mi,final_f1ma))
    #     else:
    #         print('test F1Mi : {}, test F1Ma : {}'.format(final_f1mi,final_f1ma))
    print('wait realise...')
    


def test_transductive(model: Model, data, LINEAR_PROBE_PARAM, device):
    model.eval()
    z = model(data.x, data.edge_index).detach()
    # label_classification(z, data.y, ratio=0.1, shuffle = True)
    linear_probe_nodeclas(z, data, LINEAR_PROBE_PARAM, device)



def Train_GRACE_nodecls(margs):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ################################################ prepare data
    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])

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

    if dataset_name == 'ogbn-arxiv':
        data = transform(dataset[0])
        split_idx = dataset.get_idx_split()
        # data.train_nodes = split_idx['train']
        # data.val_nodes = split_idx['valid']
        # data.test_nodes = split_idx['test']
        data.train_mask = index_to_mask(split_idx['train'],size=data.num_nodes)      
        data.val_mask   = index_to_mask(split_idx['valid'],size=data.num_nodes)    
        data.test_mask  = index_to_mask(split_idx['test'] ,size=data.num_nodes)    

    elif dataset_name in ['Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo']:
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data) # 这些数据集没有mask划分 划分完会有train_mask=[18333], val_mask=[18333], test_mask=[18333]  'Coauthor-CS'为例
    else:
        data = transform(dataset[0])

    
    data = data.to(device)
    ###########################################################################

    inductive_task = (margs.mode == 'inductive')

    MDT = build_easydict()
    config                = MDT['MODEL']['PARAM']
    LINEAR_PROBE_PARAM    = MDT['MODEL']['LINEAR_PROBE_PARAM']

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
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask ,data.test_mask
        train_mask = train_mask.cpu()
        val_mask   = val_mask.cpu()
        test_mask  = test_mask.cpu()

        # train_mask, test_mask, val_mask = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
        # 如果用小图做inductive任务 肯定比用transductive差 因为训练集的数据太小 其他点不可见，但transductive中全部节点可见
        kwargs = {'batch_size': batch_size}
        train_loader = NeighborLoader(data, input_nodes=train_mask,
                                    num_neighbors=[25], shuffle=True, **kwargs)
        val_loader = NeighborLoader(copy.copy(data), input_nodes=val_mask,
                                    num_neighbors=[-1], shuffle=False, **kwargs)
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_mask,
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
        test_inductive(model, val_loader, test_loader, val_mask, test_mask, data.y, device)
    else:
        test_transductive(model, data, LINEAR_PROBE_PARAM, device)


