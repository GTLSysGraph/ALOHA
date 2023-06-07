import torch
import torch.nn as nn
from utils import set_seed
import numpy as np

def train_tranductive(seeds,graph, model, device):
    g = graph
    g = g.int().to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    test_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_seed(seed)
        # model training
        print("Training...")
        train(g, features, labels, masks, model)

        # test the model
        print("Testing...")
        acc = evaluate(g, features, labels, masks[2], model)
        print("Test accuracy {:.4f}".format(acc))

        test_acc_list.append(acc)

    final_test_acc, final_test_acc_std = np.mean(test_acc_list), np.std(test_acc_list)
    print(f"# final_test_acc: {final_test_acc:.4f}±{final_test_acc_std:.4f}")



def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )