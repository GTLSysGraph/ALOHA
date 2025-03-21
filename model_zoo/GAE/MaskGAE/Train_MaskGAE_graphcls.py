from model_zoo.GAE.MaskGAE.build_easydict import *
from utils import Logger, set_seed, tab_printer
import torch
import torch.nn as nn
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from .model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from .mask import MaskEdge, MaskPath
import numpy as np
import time

def Train_MaskGAE_graphcls(margs):
    dataset_name = margs.dataset
    MDT = build_easydict_graphcls()
    args  = MDT['MODEL']['PARAM']

    args.save_path =  "./model_zoo/GAE/MaskGAE/model_save/model_graphclas_{}".format(dataset_name)
    
    args.save_path += '.pth'

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = margs.dataset
    assert margs.task == 'graph'
    assert dataset_name in ['IMDB-BINARY' ,'IMDB-MULTI' ,'PROTEINS', 'COLLAB', 'MUTAG', 'REDDIT-BINARY', 'NCI1']

    root = '/home/songsh/GCL/datasets_pyg'
    dataset = TUDataset(root, dataset_name, use_node_attr=True, use_edge_attr=True)

    max_deg = 0
    for data in dataset:
        max_deg = max(max_deg, degree(data.edge_index[1]).max().item())
        max_deg = max(max_deg, degree(data.edge_index[0]).max().item())

    loader = DataLoader(dataset, batch_size=args.graph_batch_size, shuffle=False)
    transform = [
        T.ToUndirected(),
        T.ToDevice(device),
    ]
    if not dataset_name in ['MUTAG']:
        transform.append(T.OneHotDegree(int(max_deg)))
        
    transform = T.Compose(transform)

    splits_all = []
    for data in loader:
        data = transform(data)
        train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                                                            split_labels=True, add_negative_train_samples=True)(data)
        splits = dict(train=train_data, valid=val_data, test=test_data, full=data)
        splits_all.append(splits)

    if args.mask == 'Path':
        mask = MaskPath(p=args.p, num_nodes=None, 
                        start=args.start,
                        walk_length=args.encoder_layers+1)
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

    train_linkpred(model, splits_all, args, device=device)
    # for splits in splits_all:
    #     splits.pop('train', None)
    #     splits.pop('valid', None)
    #     splits.pop('test', None)
    train_graphclas(model, splits_all, args, device=device)





def train_linkpred(model, splits_all, args, device="cpu"):

    def train(data):
        model.train()
        loss = model.train_epoch(data.to(device), optimizer,
                                 alpha=args.alpha, batch_size=args.batch_size)
        return loss

    @torch.no_grad()
    def test(splits_all, batch_size=2**16):
        model.eval()
        val_pred = []
        val_label = []
        test_pred = []
        test_label = []
        for splits in splits_all:
            z = model.encoder(splits['train'].x.to(device), splits['train'].edge_index.to(device))
            
            val_pos_pred = model.edge_decoder(z, splits['valid'].pos_edge_label_index).squeeze()
            val_pred.append(val_pos_pred)
            val_label.append(torch.ones_like(val_pos_pred))
            
            val_neg_pred = model.edge_decoder(z, splits['valid'].neg_edge_label_index).squeeze()
            val_pred.append(val_neg_pred)
            val_label.append(torch.zeros_like(val_neg_pred))
            
            test_pos_pred = model.edge_decoder(z, splits['test'].pos_edge_label_index).squeeze()
            test_pred.append(test_pos_pred)
            test_label.append(torch.ones_like(test_pos_pred))
            
            test_neg_pred = model.edge_decoder(z, splits['test'].neg_edge_label_index).squeeze()
            test_pred.append(test_neg_pred)
            test_label.append(torch.zeros_like(test_neg_pred))  
            
        val_pred = torch.cat(val_pred, dim=0).detach().cpu().numpy()
        val_label = torch.cat(val_label, dim=0).detach().cpu().numpy()
        test_pred = torch.cat(test_pred, dim=0).detach().cpu().numpy()
        test_label = torch.cat(test_label, dim=0).detach().cpu().numpy()
        
        valid_auc, valid_ap = roc_auc_score(val_label, val_pred), average_precision_score(val_label, val_pred)
        test_auc, test_ap = roc_auc_score(test_label, test_pred), average_precision_score(test_label, test_pred)

        results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
        return results

    monitor = 'AUC'
    save_path = args.save_path
    runs = 1
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
            loss = 0.
            for splits in splits_all:
                loss += train(splits['train'])
            t2 = time.time()

            if epoch % args.eval_period == 0:
                results = test(splits_all)

                valid_result = results[monitor][0]
                if valid_result > best_valid:
                    best_valid = valid_result
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)
                    cnt_wait = 0
                else:
                    cnt_wait += 1

                if args.debug:
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
                    print('#' * 10)
                if cnt_wait == args.patience:
                    print('Early stopping!')
                    break
        print('##### Testing on {}/{}'.format(run + 1, args.runs))

        model.load_state_dict(torch.load(save_path))
        results = test(splits_all)
        
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


def train_graphclas(model, splits_all, args, device='cpu'):
    def train_cls(x, y):
        clf.train()
        optimizer.zero_grad()
        logits = clf(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        
    @torch.no_grad()
    def test(x, y):
        clf.eval()
        logits = clf(x).argmax(1)
        return (logits == y).float().mean()

    graph_embeds, graph_labels = [], []
    with torch.no_grad():
        model.eval()
        for splits in splits_all:
            data = splits['full'].to(device)
            batch_embeds = model.encoder.get_graph_embedding(data.x, data.edge_index, data.batch, 
                                                             l2_normalize=args.l2_normalize, pooling=args.pooling)
            graph_embeds.append(batch_embeds)
            graph_labels.append(data.y)
        
        embeddings = torch.cat(graph_embeds, dim=0)
        labels = torch.cat(graph_labels, dim=0).squeeze()

    loss_fn = nn.CrossEntropyLoss()
    
    accuracies = []
    x, y = embeddings.cpu().numpy(), labels.cpu().numpy()

    logger = Logger(args.runs, args)

    print('Start Training (Graph Classification)...')
    for run in range(args.runs):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = nn.Linear(embeddings.size(1), labels.max().item() + 1).to(device)
            nn.init.xavier_uniform_(clf.weight.data)        
            clf.bias.data.fill_(0.)
            optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=args.graphclas_weight_decay)

            x_train, y_train = torch.from_numpy(x_train).to(embeddings.device), torch.from_numpy(y_train).to(embeddings.device)
            x_test, y_test = torch.from_numpy(x_test).to(embeddings.device), torch.from_numpy(y_test).to(embeddings.device)

            best_acc = 0.
            best_epoch = 0
            for epoch in range(1, 501):
                train_cls(x_train, y_train)
    #             acc = eval_cls(x_test, y_test).item()
    #             if acc > best_acc:
    #                 best_acc = acc
    #                 best_epoch = _
            acc = test(x_test, y_test).item()
            accuracies.append(acc)
        test_acc_mean, test_acc_std = np.mean(accuracies), np.std(accuracies)

        print(f"Run {run+1}: Best test accuray {test_acc_mean:.2%}.")
        logger.add_result(run, (test_acc_mean, test_acc_mean))

    print('##### Final Testing result (Graph Classification)')
    logger.print_statistics()