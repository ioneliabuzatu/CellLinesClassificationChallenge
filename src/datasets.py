import os
from typing import Dict, Optional, Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=25),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
])


class TrainDataset(Dataset):
    def __init__(self,
                 dataset: np.ndarray,
                 label_encoding: Dict[str, Optional[Any]],
                 transform: transforms.Compose
                 ):
        self.dataset = dataset
        self.label_encoding = label_encoding
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx, 0]
        label = self.label_encoding[self.dataset[idx, 1]]

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        return img, label


class TestDataloader(Dataset):
    def __init__(self, img_dir: str, transform: transforms.Compose):
        self.img_dir = img_dir
        self.images = [file for file in os.listdir(img_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        img_id = int(file_name.split('.')[0])
        with open(os.path.join(self.img_dir, file_name), 'rb') as arr:
            img = np.load(arr)
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, img_id


class InferenceDataset(Dataset):
    def __init__(self, inputs, img_ids, transform: transforms.Compose):
        self.inputs = inputs
        self.img_ids = img_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.inputs[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, img_id
