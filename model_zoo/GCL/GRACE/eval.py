import numpy as np
import functools
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from torch.utils.data import DataLoader
from utils import Logger
import time 

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


@repeat(3)
def label_classification(embeddings, y, ratio,shuffle):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio,shuffle= shuffle)
    
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    
    return {
        'F1Mi': micro,
        'F1Ma': macro
    }



def label_classification_no_repeat(embeddings, y, ratio,shuffle):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio,shuffle= shuffle)
    
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    
    return {
        'F1Mi': micro,
        'F1Ma': macro
    }







def linear_probe_nodeclas(embeddings, data, args, device='cpu'):
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
    
    embedding = embeddings

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
