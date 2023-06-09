import torch
from     easydict        import EasyDict
from     datasets_dgl.data_dgl import *
from     sklearn.metrics import average_precision_score, roc_auc_score
from     model_zoo.GAE.DGL_VGAE.build_easydict import *
from     model_zoo.GAE.DGL_VGAE.preprocessed import *
from     model_zoo.GAE.DGL_VGAE.model import *
import argparse
import os
import time
import dgl
import numpy as np
import scipy.sparse as sp
from dgl.data import CoraGraphDataset


def Train_VGAE_linkcls(margs):
    # preprocessed
    #######################
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = margs.dataset
    DATASET = EasyDict()
    if dataset_name.split('-')[0] == 'Attack':
        # dataset_name = dataset_name.split('-')[1]
        DATASET.ATTACK = EasyDict()
        DATASET.ATTACK.PARAM = {
            "data":dataset_name,
            "attack":margs.attack.split('-')[0],
            "ptb_rate":margs.attack.split('-')[1]
        }
        # now just attack use
        dataset  = load_data(DATASET['ATTACK']['PARAM'])
        graph = dataset.graph
    else:
        DATASET.PARAM = {
            "data":dataset_name,
        }
        dataset  = load_data(DATASET['PARAM'])
        graph = dataset[0]


    num_features = graph.ndata['feat'].shape[1]

    MDT = build_easydict()
    args  = MDT['MODEL']['PARAM']
    vgae_model = VGAEModel(num_features, args.hidden1, args.hidden2, device)
    Train_VGAE_linkcls_usemd(args, vgae_model, graph, device)





def Train_VGAE_linkcls_usemd(args, model, graph, device):
    # Extract node features
    feats = graph.ndata["feat"].to(device)

    # generate input
    adj_orig = graph.adjacency_matrix().to_dense()
  
    # build test set with 10% positive links
    (
        train_edge_idx,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges_dgl(graph, adj_orig)

    graph = graph.to(device)

    # create train graph
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx,preserve_nodes=True)
    train_graph = train_graph.to(device)
    adj = train_graph.adjacency_matrix().to_dense().to(device)

    # compute loss parameters
    weight_tensor, norm = compute_loss_para(adj,device)

    # create model
    # vgae_model = model(in_dim, args.hidden1, args.hidden2, device)
    vgae_model = model.to(device)

    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print(
        "Total Parameters:",
        sum([p.nelement() for p in vgae_model.parameters()]),
    )

    
    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits = vgae_model.forward(graph, feats)

        # compute loss
        loss = norm * F.binary_cross_entropy(
            logits.view(-1), adj.view(-1), weight=weight_tensor
        )
        kl_divergence = (0.5/ logits.size(0)
            * (
                1
                + 2 * vgae_model.log_std
                - vgae_model.mean**2
                - torch.exp(vgae_model.log_std) ** 2
            )
            .sum(1)
            .mean()
        )
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        print(
            "Epoch:",
            "%04d" % (epoch + 1),
            "train_loss=",
            "{:.5f}".format(loss.item()),
            "train_acc=",
            "{:.5f}".format(train_acc),
            "val_roc=",
            "{:.5f}".format(val_roc),
            "val_ap=",
            "{:.5f}".format(val_ap),
            "time=",
            "{:.5f}".format(time.time() - t),
        )

    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    # roc_means.append(test_roc)
    # ap_means.append(test_ap)
    print(
        "End of training!",
        "test_roc=",
        "{:.5f}".format(test_roc),
        "test_ap=",
        "{:.5f}".format(test_ap),
    )



def compute_loss_para(adj,device):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score



