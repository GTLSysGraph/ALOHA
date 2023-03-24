import os.path as osp
import time
import argparse

from datasets_pyg.data_pyg import get_dataset
from model_zoo.GAE.MaskGAE.build_easydict import *

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
from utils import Logger, set_seed, tab_printer

from model_zoo.GAE.MaskGAE.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from model_zoo.GAE.MaskGAE.mask import MaskEdge, MaskPath


def train_linkpred(model, splits, args, device="cpu"):

    def train(data):
        model.train()
        loss = model.train_epoch(data.to(device), optimizer,
                                 alpha=args.alpha, batch_size=args.batch_size)
        return loss

    @torch.no_grad()
    def test(splits, batch_size=2**16):
        model.eval()
        train_data = splits['train'].to(device)
        z = model(train_data.x, train_data.edge_index)

        valid_auc, valid_ap = model.test(
            z, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index, batch_size=batch_size)

        test_auc, test_ap = model.test(
            z, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index, batch_size=batch_size)

        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results


    monitor = 'AUC'
    save_path = args.save_path
    runs = args.runs # 1
    loggers = {
        'AUC': Logger(runs, args),
        'AP': Logger(runs, args),
    }
    print('Start Training (Link Prediction Pretext Training)...')
    for run in range(runs):
        model.reset_parameters()

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

        best_valid = 0.0
        best_epoch = 0
        cnt_wait = 0
        for epoch in range(1, 1 + args.epochs):

            t1 = time.time()
            loss = train(splits['train'])
            t2 = time.time()

            if epoch % args.eval_period == 0:
                results = test(splits)

                valid_result = results[monitor][0]
                if valid_result > best_valid:
                    best_valid = valid_result
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                    cnt_wait = 0
                else:
                    cnt_wait += 1

                for key, result in results.items():
                    valid_result, test_result = result
                    print(key)
                    print(f'Run: {run + 1:02d} / {args.runs:02d}, '
                          f'Epoch: {epoch:02d} / {args.epochs:02d}, '
                          f'Best_epoch: {best_epoch:02d}, '
                          f'Best_valid: {best_valid:.2%}%, '
                          f'Loss: {loss:.4f}, '
                          f'Valid: {valid_result:.2%}, '
                          f'Test: {test_result:.2%}',
                          f'Training Time/epoch: {t2-t1:.3f}')
                print('#' * round(140*epoch/(args.epochs+1)))
                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break
        print('##### Testing on {}/{}'.format(run + 1, args.runs))

        model.load_state_dict(torch.load(save_path))
        results = test(splits, model)

        for key, result in results.items():
            valid_result, test_result = result
            print(key)
            print(f'**** Testing on Run: {run + 1:02d}, '
                  f'Best Epoch: {best_epoch:02d}, '
                  f'Valid: {valid_result:.2%}, '
                  f'Test: {test_result:.2%}')

        for key, result in results.items():
            loggers[key].add_result(run, result)

    print('##### Final Testing result (Link Prediction Pretext Training)')
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


def train_nodeclas(model, data, args, device='cpu'):
    def train(loader):
        clf.train()
        for nodes in loader:
            optimizer.zero_grad()
            loss_fn(clf(embedding[nodes]), y[nodes]).backward()
            optimizer.step()

    @torch.no_grad()
    def test(loader):
        clf.eval()
        logits = []
        labels = []
        for nodes in loader:
            logits.append(clf(embedding[nodes]))
            labels.append(y[nodes])
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0).cpu()
        logits = logits.argmax(1)
        return (logits == labels).float().mean().item()

    if hasattr(data, 'train_mask'):
        train_loader = DataLoader(data.train_mask.nonzero().squeeze(), pin_memory=False, batch_size=512, shuffle=True)
        test_loader = DataLoader(data.test_mask.nonzero().squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
        val_loader = DataLoader(data.val_mask.nonzero().squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
    else:
        train_loader = DataLoader(data.train_nodes.squeeze(), pin_memory=False, batch_size=4096, shuffle=True)
        test_loader = DataLoader(data.test_nodes.squeeze(), pin_memory=False, batch_size=20000, shuffle=False)
        val_loader = DataLoader(data.val_nodes.squeeze(), pin_memory=False, batch_size=20000, shuffle=False)

    data = data.to(device)
    y = data.y.squeeze()
    embedding = model.encoder.get_embedding(data.x, data.edge_index, l2_normalize=args.l2_normalize)

    loss_fn = nn.CrossEntropyLoss()
    clf = nn.Linear(embedding.size(1), y.max().item() + 1).to(device)

    logger = Logger(args.runs, args)

    print('Start Training (Node Classification)...')
    for run in range(args.runs):
        nn.init.xavier_uniform_(clf.weight.data)
        nn.init.zeros_(clf.bias.data)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=args.nodeclas_weight_decay)  # 1 for citeseer

        best_val_metric = test_metric = 0
        start = time.time()
        for epoch in range(1, 101):
            train(train_loader)
            val_metric, test_metric = test(val_loader), test(test_loader)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
            end = time.time()
            if args.debug:
                print(f"Epoch {epoch:02d} / {100:02d}, Valid: {val_metric:.2%}, Test {test_metric:.2%}, Best {best_test_metric:.2%}, Time elapsed {end-start:.4f}")

        print(f"Run {run+1}: Best test accuray {best_test_metric:.2%}.")
        logger.add_result(run, (best_val_metric, best_test_metric))

    print('##### Final Testing result (Node Classification)')
    logger.print_statistics()




def Train_MaskGAE_nodecls(margs):
    dataset_name = margs.dataset
    MDT = build_easydict_nodecls()
    args         = MDT['MODEL']['PARAM']
    
    # if not args.save_path.endswith('.pth'):
    #     args.save_path += '.pth'
    if dataset_name.split('-')[0] == 'Attack':
        # args.save_path = 'model_nodeclas'
        args.save_path = "./model_zoo/MaskGAE/model_save/model_nodeclas_{}_{}_{}".format(dataset_name.split('-')[1].lower(),margs.attack.split('-')[0],margs.attack.split('-')[1])
    else:
        args.save_path =  "./model_zoo/MaskGAE/model_save/model_nodeclas_{}".format(dataset_name.lower())
    
    args.save_path += '.pth'

    set_seed(args.seed)
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.ToUndirected(),
        T.ToDevice(device),
    ])


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
        data.train_nodes = split_idx['train']
        data.val_nodes = split_idx['valid']
        data.test_nodes = split_idx['test']
    elif dataset_name in ['Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo']:
        data = transform(dataset[0])
        data = T.RandomNodeSplit(num_val=0.1, num_test=0.8)(data) # 这些数据集没有mask划分
    else:
        data = transform(dataset[0])

    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.1, num_test=0.05,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=True)(data)
    # phy example Data(x=[34493, 8415], edge_index=[2, 421536], y=[34493], train_mask=[34493], val_mask=[34493], test_mask=[34493], pos_edge_label=[210768], pos_edge_label_index=[2, 210768], neg_edge_label=[210768], neg_edge_label_index=[2, 210768]

    splits = dict(train=train_data, valid=val_data, test=test_data)

    if args.mask == 'Path':
        mask = MaskPath(p=args.p, num_nodes=data.num_nodes, 
                        start=args.start,
                        walk_length=3)
    elif args.mask == 'Edge':
        mask = MaskEdge(p=args.p)
    else:
        mask = None

    
    encoder = GNNEncoder(data.num_features, args.encoder_channels, args.hidden_channels,
                     num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                     bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                            num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)


    model = MaskGAE(encoder, edge_decoder, degree_decoder, mask).to(device)

    print(model)

    train_linkpred(model, splits, args, device=device)
    train_nodeclas(model, data, args, device=device)