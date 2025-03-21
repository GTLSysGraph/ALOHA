import dgl
import torch
import torch as th
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from .InfoGraph.evaluate_embedding import evaluate_embedding
from .InfoGraph.model import InfoGraph
from .build_easydict import build_InfoGraph_graphcls
from datasets_dgl.data_dgl import load_graph_data

def collate(samples):
    """collate function for building graph dataloader"""

    graphs, labels = map(list, zip(*samples))

    # generate batched graphs and labels
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)

    n_graphs = len(graphs)
    graph_id = th.arange(n_graphs)
    graph_id = dgl.broadcast_nodes(batched_graph, graph_id)

    batched_graph.ndata["graph_id"] = graph_id

    return batched_graph, batched_labels





def Train_InfoGraph_graphcls(margs):
    #############################################################################################

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = margs.dataset
    assert margs.task == 'graph'
    assert dataset_name in ['IMDB-BINARY' ,'IMDB-MULTI' ,'PROTEINS', 'COLLAB', 'MUTAG', 'REDDIT-BINARY', 'NCI1']

    MDT = build_InfoGraph_graphcls()
    args  = MDT['MODEL']['PARAM']

    # load dataset from dgl.data.GINDataset
    dataset = GINDataset(dataset_name, self_loop=False)

    # get graphs and labels
    graphs, labels = map(list, zip(*dataset))

    # generate a full-graph with all examples for evaluation
    wholegraph = dgl.batch(graphs)
    wholegraph.ndata["attr"] = wholegraph.ndata["attr"].to(th.float32)

    # create dataloader for batch training
    dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        drop_last=False,
        shuffle=True,
    )



    in_dim = wholegraph.ndata["attr"].shape[1]

    # Step 2: Create model =================================================================== #
    model = InfoGraph(in_dim, args.hid_dim, args.n_layers)
    model = model.to(device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    print("===== Before training ======")

    wholegraph = wholegraph.to(device)
    wholefeat = wholegraph.ndata["attr"]

    emb = model.get_embedding(wholegraph, wholefeat).cpu()
    res = evaluate_embedding(emb, labels, device)

    """ Evaluate the initialized embeddings """
    """ using logistic regression and SVM(non-linear) """
    print("logreg {:4f}, svc {:4f}".format(res[0], res[1]))

    best_logreg = 0
    best_logreg_epoch = 0
    best_svc = 0
    best_svc_epoch = 0

    # Step 4: training epochs =============================================================== #
    for epoch in range(args.epochs):
        loss_all = 0
        model.train()

        for graph, label in dataloader:
            graph = graph.to(device)
            feat = graph.ndata["attr"]
            graph_id = graph.ndata["graph_id"]

            n_graph = label.shape[0]

            optimizer.zero_grad()
            loss = model(graph, feat, graph_id)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        print("Epoch {}, Loss {:.4f}".format(epoch, loss_all))

        if epoch % args.log_interval == 0:
            # evaluate embeddings
            model.eval()
            emb = model.get_embedding(wholegraph, wholefeat).cpu()
            res = evaluate_embedding(emb, labels, device)

            if res[0] > best_logreg:
                best_logreg = res[0]
                best_logreg_epoch = epoch

            if res[1] > best_svc:
                best_svc = res[1]
                best_svc_epoch = epoch

            print(
                "best logreg {:4f}, epoch {} | best svc: {:4f}, epoch {}".format(
                    best_logreg, best_logreg_epoch, best_svc, best_svc_epoch
                )
            )

    print("Training End")
    print("best logreg {:4f} ,best svc {:4f}".format(best_logreg, best_svc))

