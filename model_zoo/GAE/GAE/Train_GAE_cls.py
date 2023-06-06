import torch
from datasets_pyg.data_pyg import get_dataset
from model_zoo.GAE.GAE.build_easydict import *
from utils import Logger, set_seed
import os.path as osp
import torch_geometric.transforms as T
from .models import *
from torch.utils.data import DataLoader
import torch.nn as nn
import time


def Train_GAE_cls(margs):

    dataset_name = margs.dataset
    MDT = build_easydict()
    args              = MDT['MODEL']['PARAM']
    if margs.model_name == 'VGAE':
        assert args.variational == True
    else:
        assert args.variational == False

    args_linear_probe = MDT['MODEL']['LP_PARAM']


    set_seed(args.seed)
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"


    path = osp.expanduser('/home/songsh/GCL/datasets_pyg')
    if dataset_name.split('-')[0] == 'Attack':
        attackmethod = margs.attack.split('-')[0]
        attackptb    = margs.attack.split('-')[1]
        path = osp.expanduser('/home/songsh/GCL/datasets_pyg/Attack_data')
        dataset = get_dataset(path, dataset_name, attackmethod, attackptb)
    else:
        path = osp.join(path, dataset_name)
        dataset = get_dataset(path, dataset_name)


    transform4lcls = T.Compose([
            T.NormalizeFeatures(),
            T.ToDevice(device),
            T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                            split_labels=True, add_negative_train_samples=False),
    ])

    train_data, val_data, test_data = transform4lcls(dataset[0])

    in_channels, out_channels = dataset.num_features, 256

    if not args.variational and not args.linear:
        model = GAE(GCNEncoder(in_channels, out_channels))
    elif not args.variational and args.linear:
        model = GAE(LinearEncoder(in_channels, out_channels))
    elif args.variational and not args.linear:
        model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    elif args.variational and args.linear:
        model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_linkcls(args, model, optimizer,train_data, val_data, test_data)

    if margs.task == 'node':
        transform4ncls = T.Compose([
            T.ToUndirected(),
            T.ToDevice(device),
        ])
        data = transform4ncls(dataset[0])
        train_nodeclas(model, data, args_linear_probe, device=device)












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
    # 这里的model要eval
    with torch.no_grad():
        embedding = model.encode(data.x, data.edge_index)

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









def train_linkcls(args, model, optimizer,train_data, val_data, test_data):
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        if args.variational:
            loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test(data):
        model.eval()
        z = model.encode(data.x, data.edge_index)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    

    for epoch in range(1, args.epochs + 1):
        loss = train()
        auc, ap = test(test_data)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    

