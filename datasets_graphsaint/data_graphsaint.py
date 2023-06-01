from sklearn.metrics import f1_score
import torch
import os
import numpy as np
import random
from functools import namedtuple
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import json
import dgl

# load data of GraphSAINT and convert them to the format of dgl
# 来自 https://github.com/lt610/GraphSaint/blob/master/train_sampling.py graphsaint的pytorch实现中用到的数据集 
# 在graphsaint的官方代码中提供了下载的云盘 https://github.com/GraphSAINT/GraphSAINT
# 自己在网盘中已经进行了保存

def load_GraphSAINT_data(dstname, multilabel):
    prefix = "/home/songsh/GCL/datasets_graphsaint/{}".format(dstname)
    DataType = namedtuple('Dataset', ['num_classes', 'train_nid', 'g'])

    adj_full = scipy.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(np.bool)
    g = dgl.from_scipy(adj_full)
    num_nodes = g.num_nodes()

    adj_train = scipy.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(np.bool)
    train_nid = np.array(list(set(adj_train.nonzero()[0])))

    role = json.load(open('{}/role.json'.format(prefix)))
    mask = np.zeros((num_nodes,), dtype=bool)
    train_mask = mask.copy()
    train_mask[role['tr']] = True
    val_mask = mask.copy()
    val_mask[role['va']] = True
    test_mask = mask.copy()
    test_mask[role['te']] = True

    feats = np.load('{}/feats.npy'.format(prefix))
    scaler = StandardScaler()
    scaler.fit(feats[train_nid])
    feats = scaler.transform(feats)

    class_map = json.load(open('{}/class_map.json'.format(prefix)))
    class_map = {int(k): v for k, v in class_map.items()}
    if multilabel:
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_nodes, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_nodes,))
        for k, v in class_map.items():
            class_arr[k] = v

    g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    g.ndata['label'] = torch.tensor(class_arr, dtype=torch.float if multilabel else torch.long)
    g.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    g.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
    g.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

    data = DataType(g=g, num_classes=num_classes, train_nid=train_nid)
    return data