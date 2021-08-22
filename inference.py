import os
from time import time
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import models

import config
from datasets import predict_dataloader, transform_test


def predict(testloader: DataLoader, model: torchvision.models, device: torch.device,
            decode: Dict[int, str]) -> pd.DataFrame:
    model.eval()
    pred = []
    ids = []
    start = time()
    with torch.no_grad():
        for i, (inputs, img_id) in enumerate(testloader, 1):
            inputs = inputs.to(device)
            output = model.forward(inputs)
            for j, val in enumerate(output):
                pred.append(decode[(torch.argmax(val).item())])
                ids.append(img_id[j].item())
    return pd.DataFrame({'file_id': ids, 'cell_line': pred})


if __name__ == "__main__":
    encode = {'A549': 0, 'CACO-2': 1, 'HEK 293': 2, 'HeLa': 3, 'MCF7': 4, 'PC-3': 5, 'RT4': 6, 'U-2 OS': 7,
              'U-251 MG': 8}
    decode = {v: k for k, v in encode.items()}
    b = []
    for img_idx in range(9633, 16502):
        img_filepath = f"{config.root_all_data}/rgb-test-images/{img_idx}.png"
        assert os.path.exists(img_filepath)
        b.append(cv2.imread(img_filepath))
    c = np.arange(9633, 16502)
    test_set = predict_dataloader(inputs=b, img_ids=c, transform=transform_test)
    test_loader = DataLoader(test_set, **{"batch_size": 64, "shuffle": False, "num_workers": 20})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    n_inputs = model.fc.in_features
    classifier = nn.Linear(n_inputs, 9)
    model.fc = classifier
    model.load_state_dict(torch.load("cell_lines.ckpt"))
    model.to(device)
    predictions = predict(testloader=test_loader, model=model, device=device, decode=decode)
    predictions.to_csv('server_predictions.csv', index=False)
