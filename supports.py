from time import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader


def train(trainloader: DataLoader,
          model: torchvision.models,
          device,
          criterion: nn.CrossEntropyLoss,
          optimizer,
          verbose=20):
    model.train()
    true = []
    pred = []
    train_loss = []
    start = time()
    for i, (inputs, labels) in enumerate(trainloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        for j, val in enumerate(output):
            true.append(labels[j].item())
            pred.append(torch.argmax(val).item())
        if i % verbose == 0:
            print(f"Batch {i} of {len(trainloader)} | Train Loss: {np.mean(train_loss):.3f}")
    balanced_acc = balanced_accuracy_score(true, pred)
    return np.mean(train_loss), balanced_acc


def validate(valloader: DataLoader,
             model: torchvision.models,
             device: torch.device,
             criterion: torch.optim,
             verbose: int):
    model.eval()
    true = []
    pred = []
    val_loss = []
    start = time()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valloader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            loss = criterion(output, labels)
            val_loss.append(loss.item())
            for j, val in enumerate(output):
                true.append(labels[j].item())
                pred.append(torch.argmax(val).item())

            if i % verbose == 0:
                print(f"Batch {i} of {len(valloader)} | Val Loss: {np.mean(val_loss):.3f}")
    balanced_acc = balanced_accuracy_score(true, pred)
    return np.mean(val_loss), balanced_acc
