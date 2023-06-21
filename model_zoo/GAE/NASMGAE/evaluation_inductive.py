import torch
import torch.nn as nn
from .utils import create_optimizer, accuracy
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import copy

def evaluete(model, loaders, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob=True, mute=False):
    model.eval()
    if linear_prob:
        if len(loaders[0]) > 1:
            x_all = {"train": [], "val": [], "test": []}
            y_all = {"train": [], "val": [], "test": []}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        x_all[key].append(x)
                        y_all[key].append(subgraph.ndata["label"])  
            in_dim = x_all["train"][0].shape[1]
            encoder = LogisticRegression(in_dim, num_classes)
            num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
            if not mute:
                print(f"num parameters for finetuning: {sum(num_finetune_params)}")
                # torch.save(x.cpu(), "feat.pt")
            
            encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
            final_acc, estp_acc = mutli_graph_linear_evaluation(encoder, x_all, y_all, optimizer_f, max_epoch_f, device, mute)
            return final_acc, estp_acc
        else:
            x_all = {"train": None, "val": None, "test": None}
            y_all = {"train": None, "val": None, "test": None}

            with torch.no_grad():
                for key, loader in zip(["train", "val", "test"], loaders):
                    for subgraph in loader:
                        subgraph = subgraph.to(device)
                        feat = subgraph.ndata["feat"]
                        x = model.embed(subgraph, feat)
                        mask = subgraph.ndata[f"{key}_mask"]
                        x_all[key] = x[mask]
                        y_all[key] = subgraph.ndata["label"][mask]  
            in_dim = x_all["train"].shape[1]
            
            encoder = LogisticRegression(in_dim, num_classes)
            encoder = encoder.to(device)
            optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)

            x = torch.cat(list(x_all.values()))
            y = torch.cat(list(y_all.values()))
            num_train, num_val, num_test = [x.shape[0] for x in x_all.values()]
            num_nodes = num_train + num_val + num_test
            train_mask = torch.arange(num_train, device=device)
            val_mask = torch.arange(num_train, num_train + num_val, device=device)
            test_mask = torch.arange(num_train + num_val, num_nodes, device=device)
            
            final_acc, estp_acc = linear_probing_for_inductive_node_classiifcation(encoder, x, y, (train_mask, val_mask, test_mask), optimizer_f, max_epoch_f, device, mute)
            return final_acc, estp_acc
    else:
        raise NotImplementedError




def mutli_graph_linear_evaluation(model, feat, labels, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_val_epoch = 0
    best_val_test_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        for x, y in zip(feat["train"], labels["train"]):
            out = model(None, x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = []
            test_out = []
            for x, y in zip(feat["val"], labels["val"]):
                val_pred = model(None, x)
                val_out.append(val_pred)
            val_out = torch.cat(val_out, dim=0).cpu().numpy()
            val_label = torch.cat(labels["val"], dim=0).cpu().numpy()
            val_out = np.where(val_out >= 0, 1, 0)

            for x, y in zip(feat["test"], labels["test"]):
                test_pred = model(None, x)# 
                test_out.append(test_pred)
            test_out = torch.cat(test_out, dim=0).cpu().numpy()
            test_label = torch.cat(labels["test"], dim=0).cpu().numpy()
            test_out = np.where(test_out >= 0, 1, 0)

            val_acc = f1_score(val_label, val_out, average="micro")
            test_acc = f1_score(test_label, test_out, average="micro")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_test_acc = test_acc

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc}, test_acc:{test_acc: .4f}")

    if mute:
        print(f"# IGNORE: --- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f},  Final-TestAcc: {test_acc:.4f}--- ")
    else:
        print(f"--- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f}, Final-TestAcc: {test_acc:.4f} --- ")

    return test_acc, best_val_test_acc



def linear_probing_for_inductive_node_classiifcation(model, x, labels, mask, optimizer, max_epoch, device, mute=False):
    if len(labels.shape) > 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    train_mask, val_mask, test_mask = mask

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)  

        best_val_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(None, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(None, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(None, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    if mute:
        print(f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} ")
    else:
        print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}")

    return test_acc, estp_test_acc


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits

