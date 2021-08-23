import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

import config
import preprocessing
from datasets import train_dataloader, transform_train, transform_test
from supports import validate, train

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    writer = config.tensorboard
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    cell_lines = preprocessing.CellLinesMicroscopyData()

    train_with_labels = cell_lines.train_data_with_labels()
    X = train_with_labels[:, 1]
    y = train_with_labels[:, 2]

    train_indices, test_indices = preprocessing.CellLinesMicroscopyData().train_and_validation_indices(X, y)

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]


    train_data = np.array([X_train, y_train]).T
    test_data = np.array([X_test, y_test]).T
    # tmp_train_data = np.array(train_data)
    # test_data = pd.DataFrame(test_data)
    # test_data = test_data.T
    # train_data = pd.DataFrame(train_data)
    # train_data = train_data.T

    train_set = train_dataloader(df=train_data, label_encoding=config.encode, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=20)
    val_set = train_dataloader(df=test_data, label_encoding=config.encode, transform=transform_test)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    n_inputs = model.fc.in_features
    classifier = nn.Linear(n_inputs, 9)
    model.fc = classifier
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)
    val_loss, val_acc = validate(valloader=val_loader, model=model, device=device, criterion=criterion, verbose=2)
    writer.add_scalar('Loss/val', val_loss, 0)
    writer.add_scalar('Accuracy/val', val_acc, 0)
    og_loss = 100
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train(
            trainloader=train_loader,
            model=model,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            verbose=10
        )
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        print("Training done, now validation")
        val_loss, val_acc = validate(valloader=val_loader, model=model, device=device, criterion=criterion, verbose=10)
        print(f"Epoch {epoch}: Train Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.3f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        if val_loss < og_loss:
            torch.save(model.state_dict(), "cell_lines.ckpt")
        print(f"\n *** Model of epoch {epoch} saved ***\n")
        og_loss = val_loss
