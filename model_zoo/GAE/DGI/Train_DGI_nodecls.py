
from     datasets_dgl.data_dgl import *
from     datasets_graphsaint.data_graphsaint import *
from     easydict        import EasyDict
import   torch
from .dgi import DGI,Classifier
from .build_easydict import *
import torch.nn as nn
import time
import torch.nn.functional as F

def Train_DGI_nodecls(margs):
    #########################
    if margs.gpu_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{margs.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    multilabel = False

    dataset_name = margs.dataset
    if margs.mode in ['tranductive' , 'mini_batch']:
        if dataset_name.split('-')[0] == 'Attack':
            # dataset_name = dataset_name.split('-')[1]
            DATASET = EasyDict()
            DATASET.ATTACK = {
                "data":dataset_name,
                "attack":margs.attack.split('-')[0],
                "ptb_rate":margs.attack.split('-')[1]
            }
            # now just attack use
            dataset  = load_attack_data(DATASET['ATTACK'])
            graph = dataset.graph    
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph) # graphmae + self loop这结果也太好了，分析一下，有点意思
        else:
            if dataset_name in ['Cora','Pubmed','Citeseer']:
                dataset  = load_data(dataset_name)
                graph = dataset[0]
            elif dataset_name in ['ogbn-arxiv','ogbn-arxiv_undirected','reddit','ppi','yelp', 'amazon']:   
                multilabel_data = set(['ppi', 'yelp', 'amazon'])
                multilabel = dataset_name in multilabel_data

                dataset  = load_GraphSAINT_data(dataset_name, multilabel)
                graph = dataset.g

            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
        
        num_classes = dataset.num_classes
        num_features = graph.ndata['feat'].shape[1]
    else:
        raise Exception('Unknown mode!')

    ##########################
    g = graph
    
    features = g.ndata["feat"].to(device)
    labels = g.ndata["label"].to(device)
    if hasattr(torch, "BoolTensor"):
        train_mask = torch.BoolTensor(g.ndata["train_mask"]).to(device)
        val_mask = torch.BoolTensor(g.ndata["val_mask"]).to(device)
        test_mask = torch.BoolTensor(g.ndata["test_mask"]).to(device)
    else:
        train_mask = torch.ByteTensor(g.ndata["train_mask"]).to(device)
        val_mask = torch.ByteTensor(g.ndata["val_mask"]).to(device)
        test_mask = torch.ByteTensor(g.ndata["test_mask"]).to(device)
    
    g = g.to(device)
    n_edges = g.num_edges()

    MDT = build_easydict_nodecls()
    param         = MDT['MODEL']['PARAM']
    

    # create DGI model
    dgi = DGI(
        g,
        num_features,
        param.n_hidden,
        param.n_layers,
        nn.PReLU(param.n_hidden),
        param.dropout,
    ).to(device)

    dgi_optimizer = torch.optim.Adam(
        dgi.parameters(), lr=param.dgi_lr, weight_decay=param.weight_decay
    )

     # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    for epoch in range(param.n_dgi_epochs):
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = dgi(features)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), "/home/songsh/GCL/model_zoo/GAE/DGI/model_saved/best_dgi.pkl")
        else:
            cnt_wait += 1

        if cnt_wait == param.patience:
            print("Early stopping!")
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
            "ETputs(KTEPS) {:.2f}".format(
                epoch, np.mean(dur), loss.item(), n_edges / np.mean(dur) / 1000
            )
        )

    # create classifier model
    # 注意classifier里有个log_softmax  CrossEntropyLoss=LogSoftmax+NLLLoss 如果多标签分类需要把log_softmax去掉
    classifier = Classifier(param.n_hidden, num_classes).to(device)


    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=param.classifier_lr,
        weight_decay=param.weight_decay,
    )

    print("")

    # train classifier
    print("Loading {}th epoch".format(best_t))
    dgi.load_state_dict(torch.load("/home/songsh/GCL/model_zoo/GAE/DGI/model_saved/best_dgi.pkl"))
    embeds = dgi.encoder(features, corrupt=False)
    embeds = embeds.detach()

    if multilabel == True:
        loss_method = F.binary_cross_entropy_with_logits
    else:
        loss_method = F.cross_entropy

    dur = []
    for epoch in range(param.n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = loss_method(preds[train_mask], labels[train_mask])
        loss.backward()
        classifier_optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if multilabel == True:
            f1_val_mic, f1_val_mac = evaluate(classifier,embeds,labels,val_mask, True)
            print("val F1-mic {:.4f} | val F1-mac {:.4f} | Loss {:.4f}".format(f1_val_mic, f1_val_mac,loss))
        else:
            acc = acc_evalute(classifier, embeds, labels, val_mask)
            print(
                "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(
                    epoch,
                    np.mean(dur),
                    loss.item(),
                    acc,
                    n_edges / np.mean(dur) / 1000,
                )
            )

    if multilabel == True:
        f1_test_mic, f1_test_mac = evaluate(classifier,embeds,labels,test_mask, True)
        print("Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(f1_test_mic, f1_test_mac))
    else:
        acc = acc_evalute(classifier, embeds, labels, test_mask)
        print("Test Accuracy {:.4f}".format(acc))






def acc_evalute(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = torch.log_softmax(logits, dim=-1)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    

def calc_f1(y_true, y_pred, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
        f1_score(y_true, y_pred, average="macro")


def evaluate(model, embeds, labels, mask, multilabel=False):
    model.eval()
    with torch.no_grad():
        logits = model(embeds)
        logits = logits[mask]
        labels = labels[mask]
        f1_mic, f1_mac = calc_f1(labels.cpu().numpy(),
                                 logits.cpu().numpy(), multilabel)
        return f1_mic, f1_mac