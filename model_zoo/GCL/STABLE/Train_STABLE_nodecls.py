from     easydict        import EasyDict
from     datasets_dgl.data_dgl import *
from     datasets_dgl.utils import *
from     model_zoo.GCL.STABLE.build_easydict import *
from     model_zoo.GCL.STABLE.utils import *
from     model_zoo.GCL.STABLE.model import *
import   torch
from copy import deepcopy

def Train_STABLE_nodecls(margs):
    # preprocessed
    ######################################################################
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"

    dataset_name = margs.dataset
    DATASET = EasyDict()
    if dataset_name.split('-')[0] == 'Attack':
        dataset_name = dataset_name.split('-')[1]
        DATASET.ATTACK = EasyDict()
        DATASET.ATTACK.PARAM = {
            "data":dataset_name,
            "attack":margs.attack.split('-')[0],
            "ptb_rate":margs.attack.split('-')[1]
        }

        # now just attack use
        dataset  = load_data(DATASET['ATTACK']['PARAM'])
    else:
        raise Exception('Only Attack data now!') 
    graph = dataset.graph
    feat = graph.ndata['feat']

    MDT = build_easydict_nodecls()
    args   =  MDT['MODEL']['PARAM']
    #######################################################################

    # STABLE
    n_class = dataset.num_classes
    # Training parameters
    n_hidden = 16
    weight_decay = 5e-4
    lr = 1e-2
    
    features             = to_scipy(feat)
    perturbed_adj_sparse = to_scipy(graph.adj())

    print('===start preprocessing the graph===')
    if dataset_name == 'polblogs':
        args.jt = 0
    adj_pre = preprocess_adj(features, perturbed_adj_sparse,  threshold=args.jt)
    adj_delete = perturbed_adj_sparse - adj_pre
    _, features = to_tensor(perturbed_adj_sparse, features)
    print('===start getting contrastive embeddings===')
    embeds, _ = get_contrastive_emb(adj_pre, features.unsqueeze(dim=0).to_dense(), adj_delete=adj_delete,
                                    lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta)
    embeds = embeds.squeeze(dim=0)
    acc_total = []
    embeds = embeds.to('cpu')
    embeds = to_scipy(embeds)

    # prune the perturbed graph by the representations
    adj_clean = preprocess_adj(embeds, perturbed_adj_sparse, jaccard=False, threshold=args.cos)
    embeds = torch.FloatTensor(embeds.todense()).to(device)
    adj_clean = sparse_mx_to_torch_sparse_tensor(adj_clean)
    adj_clean = adj_clean.to_dense()
    features = features.to_dense()
    # labels = torch.LongTensor(labels)
    adj_clean = adj_clean.to(device)
    features = features.to(device)
    # labels = labels.to(device)
    print('===train ours on perturbed graph===')
    for run in range(10):
        adj_temp = adj_clean.clone()
        # add k new neighbors to each node
        get_reliable_neighbors(adj_temp, embeds, k=args.k, degree_threshold=args.threshold,device = device)
        model = GCN(embeds.shape[1], n_hidden, n_class)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        adj_temp = adj_new_norm(adj_temp, args.alpha, device=device)
        acc = train(DATASET['ATTACK']['PARAM'],graph, model, optimizer, adj_temp, run, embeds=embeds, device=device, verbose=False)
        acc_total.append(acc)
    print('Mean Accuracy:%f' % np.mean(acc_total))
    print('Standard Deviation:%f' % np.std(acc_total, ddof=1))


def train(args,graph, model, optim, adj, run, embeds, device, verbose=True):
    # Training parameters
    epochs = 200
    loss = nn.CrossEntropyLoss()
    # mask & label
    train_mask = graph.ndata['train_mask'].to(device)
    val_mask   = graph.ndata['val_mask'].to(device)
    test_mask  = graph.ndata['test_mask'].to(device)
    labels     = graph.ndata['label'].to(device)

    best_loss_val = 9999
    best_acc_val = 0
    for epoch in range(epochs):
        model.train()
        logits = model(adj, embeds)
        l = loss(logits[train_mask], labels[train_mask])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj, embeds, labels, val_mask)
        val_loss = loss(logits[val_mask], labels[val_mask])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc))
    model.load_state_dict(weights)
    torch.save(weights, '/home/songsh/GCL/model_zoo/GCL/STABLE/save_model/%s_%s_%s.pth' % (args.attack, args.data, args.ptb_rate))
    acc = evaluate(model, adj, embeds, labels, test_mask)
    print("Run {:02d} Test Accuracy {:.4f}".format(run, acc))
    return acc
