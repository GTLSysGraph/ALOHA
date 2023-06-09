import numpy as np
import torch as th
import torch.nn as nn
import random
import dgl
import warnings
warnings.filterwarnings('ignore')
import torch
from .node.dataset import process_dataset,process_dataset_appnp
from .node.model import MVGRL, LogReg
from .build_easydict import build_MVGRL_nodecls



# We fail to reproduce the reported accuracy on 'Cora', even with the authors' code.
# The accuracy reported by the original paper is based on fixed-sized subgraph-training.
# https://github.com/hengruizhang98/mvgrl/tree/main
# Cora的值和论文里提到的有出入 83.56000518798828 0.05477583035826683 其他的值参考论文中~ happy！

def Train_MVGRL_nodecls(margs):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = margs.dataset

    MDT = build_MVGRL_nodecls()
    args  = MDT['MODEL']['PARAM']
    
    args.device = device

    if args.use_sample == 'True':
        Train_MVGRL_sample_nodecls(args,dataset_name)
    else:
        Train_MVGRL_full_nodecls(args,dataset_name)






def Train_MVGRL_sample_nodecls(args, dataset_name):

    # Step 1: Prepare data =================================================================== #
    if dataset_name == 'Pubmed':
        graph, diff_graph, feat, label, train_idx, val_idx, test_idx, edge_weight = process_dataset_appnp(args.epsilon)
    else:
        graph, diff_graph, feat, label, train_idx, val_idx, test_idx, edge_weight = process_dataset(dataset_name, args.epsilon)
    edge_weight = th.tensor(edge_weight).float()
    graph.ndata['feat'] = feat
    diff_graph.edata['edge_weight'] = edge_weight

    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]
    edge_weight = th.tensor(edge_weight).float()

    train_idx = train_idx.to(args.device)
    val_idx = val_idx.to(args.device)
    test_idx = test_idx.to(args.device)

    n_node = graph.number_of_nodes()

    sample_size = args.sample_size

    lbl1 = th.ones(sample_size * 2)
    lbl2 = th.zeros(sample_size * 2)
    lbl = th.cat((lbl1, lbl2))
    lbl = lbl.to(args.device)

    # Step 2: Create model =================================================================== #
    model = MVGRL(n_feat, args.hid_dim)
    model = model.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    loss_fn = nn.BCEWithLogitsLoss()

    node_list = list(range(n_node))

    # Step 4: Training epochs ================================================================ #
    best = float('inf')
    cnt_wait = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        sample_idx = random.sample(node_list, sample_size)

        g = dgl.node_subgraph(graph, sample_idx)
        dg = dgl.node_subgraph(diff_graph, sample_idx)

        f = g.ndata.pop('feat')
        ew = dg.edata.pop('edge_weight')

        shuf_idx = np.random.permutation(sample_size)
        sf = f[shuf_idx, :]

        g = g.to(args.device)
        dg = dg.to(args.device)
        f = f.to(args.device)
        ew = ew.to(args.device)

        sf = sf.to(args.device)

        out = model(g, dg, f, sf, ew)
        loss = loss_fn(out, lbl)

        loss.backward()
        optimizer.step()

        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(model.state_dict(), '/home/songsh/GCL/model_zoo/GCL/MVGRL/model_saved/{}.pkl'.format(dataset_name))
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping')
            break

    model.load_state_dict(th.load('/home/songsh/GCL/model_zoo/GCL/MVGRL/model_saved/{}.pkl'.format(dataset_name)))

    graph = graph.to(args.device)
    diff_graph = diff_graph.to(args.device)
    feat = feat.to(args.device)
    edge_weight = edge_weight.to(args.device)
    embeds = model.get_embedding(graph, diff_graph, feat, edge_weight)

    train_embs = embeds[train_idx]
    test_embs = embeds[test_idx]

    label = label.to(args.device)
    train_labels = label[train_idx]
    test_labels = label[test_idx]
    accs = []

    # Step 5:  Linear evaluation ========================================================== #
    for _ in range(5):
        model = LogReg(args.hid_dim, n_classes)
        opt = th.optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.wd2)

        model = model.to(args.device)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(300):
            model.train()
            opt.zero_grad()
            logits = model(train_embs)
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

        model.eval()
        logits = model(test_embs)
        preds = th.argmax(logits, dim=1)
        acc = th.sum(preds == test_labels).float() / test_labels.shape[0]
        accs.append(acc * 100)

    accs = th.stack(accs)
    print(accs.mean().item(), accs.std().item())



def Train_MVGRL_full_nodecls(args,dataset_name):

    # Step 1: Prepare data =================================================================== #
    graph, diff_graph, feat, label, train_idx, val_idx, test_idx, edge_weight = process_dataset(dataset_name, args.epsilon)
    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    graph = graph.to(args.device)
    diff_graph = diff_graph.to(args.device)
    feat = feat.to(args.device)
    edge_weight = th.tensor(edge_weight).float().to(args.device)

    train_idx = train_idx.to(args.device)
    val_idx = val_idx.to(args.device)
    test_idx = test_idx.to(args.device)

    n_node = graph.number_of_nodes()
    lbl1 = th.ones(n_node * 2)
    lbl2 = th.zeros(n_node * 2)
    lbl = th.cat((lbl1, lbl2))

    # Step 2: Create model =================================================================== #
    model = MVGRL(n_feat, args.hid_dim)
    model = model.to(args.device)

    lbl = lbl.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    loss_fn = nn.BCEWithLogitsLoss()

    # Step 4: Training epochs ================================================================ #
    best = float('inf')
    cnt_wait = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        shuf_idx = np.random.permutation(n_node)
        shuf_feat = feat[shuf_idx, :]
        shuf_feat = shuf_feat.to(args.device)

        out = model(graph, diff_graph, feat, shuf_feat, edge_weight)
        loss = loss_fn(out, lbl)

        loss.backward()
        optimizer.step()

        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))

        if loss < best:
            best = loss
            cnt_wait = 0
            th.save(model.state_dict(), '/home/songsh/GCL/model_zoo/GCL/MVGRL/model_saved/{}.pkl'.format(dataset_name))
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping')
            break

    model.load_state_dict(th.load('/home/songsh/GCL/model_zoo/GCL/MVGRL/model_saved/{}.pkl'.format(dataset_name)))
    embeds = model.get_embedding(graph, diff_graph, feat, edge_weight)

    train_embs = embeds[train_idx]
    test_embs = embeds[test_idx]

    label = label.to(args.device)
    train_labels = label[train_idx]
    test_labels = label[test_idx]
    accs = []

    # Step 5:  Linear evaluation ========================================================== #
    for _ in range(5):
        model = LogReg(args.hid_dim, n_classes)
        opt = th.optim.Adam(model.parameters(), lr=args.lr2, weight_decay=args.wd2)

        model = model.to(args.device)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(300):
            model.train()
            opt.zero_grad()
            logits = model(train_embs)
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

        model.eval()
        logits = model(test_embs)
        preds = th.argmax(logits, dim=1)
        acc = th.sum(preds == test_labels).float() / test_labels.shape[0]
        accs.append(acc * 100)

    accs = th.stack(accs)
    print(accs.mean().item(), accs.std().item())