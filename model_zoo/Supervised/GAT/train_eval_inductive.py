
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
from  utils import set_seed

def train_inductive(seeds,
                    train_dataloader,
                    valid_dataloader, 
                    test_dataloader,
                    model,
                    device ):
    test_acc_list = []
    # model training
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_seed(seed)

        print("Training...")
        train(train_dataloader, valid_dataloader, device, model)

        # test the model
        print("Testing...")
        avg_score = evaluate_in_batches(test_dataloader, device, model)
        print("Test Accuracy (F1-score) {:.4f}".format(avg_score))
    
        test_acc_list.append(avg_score)

    final_test_acc, final_test_acc_std = np.mean(test_acc_list), np.std(test_acc_list)
    print(f"# final_test_acc: {final_test_acc:.4f}±{final_test_acc_std:.4f}")



# https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train_ppi.py
def evaluate(g, features, labels, model):
    model.eval()
    with torch.no_grad():
        output = model(g, features)
        pred = np.where(output.data.cpu().numpy() >= 0, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), pred, average="micro")
        return score


def evaluate_in_batches(dataloader, device, model):
    total_score = 0
    for batch_id, batched_graph in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata["feat"]
        labels = batched_graph.ndata["label"]
        score = evaluate(batched_graph, features, labels, model)
        total_score += score
    return total_score / (batch_id + 1)  # return average score


def train(train_dataloader, val_dataloader, device, model):
    # define loss function and optimizer
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)

    # training loop
    for epoch in range(400):
        model.train()
        logits = []
        total_loss = 0
        # mini-batch loop
        for batch_id, batched_graph in enumerate(train_dataloader):
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata["feat"].float()
            labels = batched_graph.ndata["label"].float()
            logits = model(batched_graph, features)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            "Epoch {:05d} | Loss {:.4f} |".format(
                epoch, total_loss / (batch_id + 1)
            )
        )

        if (epoch + 1) % 5 == 0:
            avg_score = evaluate_in_batches(
                val_dataloader, device, model
            )  # evaluate F1-score instead of loss
            print(
                "                            Acc. (F1-score) {:.4f} ".format(
                    avg_score
                )
            )
