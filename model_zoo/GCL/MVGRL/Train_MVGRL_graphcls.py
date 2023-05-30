import torch as th
import torch
import dgl
from dgl.dataloading import GraphDataLoader
import warnings
from .build_easydict import build_MVGRL_graphcls
from .graph.dataset import *
from .graph.utils import linearsvc
from .graph.model import MVGRL

def collate(samples):
    ''' collate function for building the graph dataloader'''
    graphs, diff_graphs, labels = map(list, zip(*samples))

    # generate batched graphs and labels
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    batched_diff_graph = dgl.batch(diff_graphs)

    n_graphs = len(graphs)
    graph_id = th.arange(n_graphs)
    graph_id = dgl.broadcast_nodes(batched_graph, graph_id)

    batched_graph.ndata['graph_id'] = graph_id

    return batched_graph, batched_diff_graph, batched_labels



def Train_MVGRL_graphcls(margs):
    #############################################################################################
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"

    dataset_name = margs.dataset
    assert margs.task == 'graph'
    assert dataset_name in ['IMDB-BINARY' ,'IMDB-MULTI' ,'PROTEINS', 'COLLAB', 'MUTAG', 'REDDIT-BINARY', 'NCI1']

    MDT = build_MVGRL_graphcls()
    args  = MDT['MODEL']['PARAM']

    # Step 1: Prepare data =================================================================== #
    dataset = load(dataset_name)

    graphs, diff_graphs, labels = map(list, zip(*dataset))
    print('Number of graphs:', len(graphs))
    # generate a full-graph with all examples for evaluation

    wholegraph = dgl.batch(graphs)
    whole_dg = dgl.batch(diff_graphs)

    # create dataloader for batch training
    dataloader = GraphDataLoader(dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate,
                                 drop_last=False,
                                 shuffle=True)
    
    
    in_dim = wholegraph.ndata['feat'].shape[1]

    # Step 2: Create model =================================================================== #
    model = MVGRL(in_dim, args.hid_dim, args.n_layers)
    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    print('===== Before training ======')

    wholegraph = wholegraph.to(device)
    whole_dg = whole_dg.to(device)
    wholefeat = wholegraph.ndata.pop('feat')
    whole_weight = whole_dg.edata.pop('edge_weight')

    embs = model.get_embedding(wholegraph, whole_dg, wholefeat, whole_weight)
    lbls = th.LongTensor(labels)
    acc_mean, acc_std = linearsvc(embs, lbls)
    print('accuracy_mean, {:.4f}'.format(acc_mean))

    best = float('inf')
    cnt_wait = 0
    # Step 4: Training epochs =============================================================== #
    for epoch in range(args.epochs):
        loss_all = 0
        model.train()

        for graph, diff_graph, label in dataloader:
            graph = graph.to(device)
            diff_graph = diff_graph.to(device)

            feat = graph.ndata['feat']
            graph_id = graph.ndata['graph_id']
            edge_weight = diff_graph.edata['edge_weight']
            n_graph = label.shape[0]

            optimizer.zero_grad()
            loss = model(graph, diff_graph, feat, edge_weight, graph_id)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()

        print('Epoch {}, Loss {:.4f}'.format(epoch, loss_all))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            th.save(model.state_dict(), f'/home/songsh/GCL/model_zoo/GCL/MVGRL/model_saved/{dataset_name}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping')
            break

    print('Training End')

    # Step 5:  Linear evaluation ========================================================== #
    model.load_state_dict(th.load(f'/home/songsh/GCL/model_zoo/GCL/MVGRL/model_saved/{dataset_name}.pkl'))
    embs = model.get_embedding(wholegraph, whole_dg, wholefeat, whole_weight)

    acc_mean, acc_std = linearsvc(embs, lbls)
    print('accuracy_mean, {:.4f}'.format(acc_mean))