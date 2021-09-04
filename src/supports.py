import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader


def train(train_dataloader: DataLoader,
          model: torchvision.models,
          device,
          criterion: nn.CrossEntropyLoss,
          optimizer,
          epoch: int,
          verbose: int = 30,
          ):
    model.train()
    true = []
    pred = []
    train_loss = []
    len_train_dataloader = len(train_dataloader)
    for i, (inputs, labels) in enumerate(train_dataloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.to(device), labels.to(device)

        def closure():
            # https://gist.github.com/manuel-delverme/8c8719b55ecf6c0b7f8dd932976ec4b9
            output = model.forward(inputs)
            loss = criterion(output, labels)
            layer_norm = torch.mean(torch.tensor([p.norm() for p in model.parameters()]))
            ineq_defect = [(layer_norm - 0.025).reshape(1, -1),]
            return loss, None, ineq_defect

        # optimizer.step(closure)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # output = model.forward(inputs)
        # loss, _, eq_defect = closure()

        train_loss.append(loss.item())
        for j, val in enumerate(output):
            true.append(labels[j].item())
            pred.append(torch.argmax(val).item())
        if i % verbose == 0:
            print(f"Epoch {epoch} | Batch {i} of {len_train_dataloader} | Train Loss: {np.mean(train_loss):.3f}")
    balanced_acc = balanced_accuracy_score(true, pred)
    return np.mean(train_loss), balanced_acc


def validate(val_dataloader: DataLoader,
             model: torchvision.models,
             device: torch.device,
             criterion: torch.optim,
             epoch: int,
             verbose: int = 30,
             ):
    model.eval()
    true = []
    pred = []
    val_loss = []
    len_val_dataloader = len(val_dataloader)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            loss = criterion(output, labels)
            val_loss.append(loss.item())
            for j, val in enumerate(output):
                true.append(labels[j].item())
                pred.append(torch.argmax(val).item())

            if i % verbose == 0:
                print(f"Epoch {epoch} | Batch {i} of {len_val_dataloader} | Val Loss: {np.mean(val_loss):.3f}")
    balanced_acc = balanced_accuracy_score(true, pred)
    return np.mean(val_loss), balanced_acc
