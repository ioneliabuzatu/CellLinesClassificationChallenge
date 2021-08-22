import json
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection

import config


class CellLinesMicroscopyData(object):
    def __init__(self):
        self.range_train_imgs = range(1, 9633)
        self.range_test_imgs = range(9633, 16502)
        self.root_all_data = config.root_all_data
        self.dir_train_images_channels = config.dir_train_images_separate_channels
        self.dir_test_images_channels = config.dir_test_images_separate_channels
        self.dir_train_images_rgb = config.dir_train_images_rgb
        self.dir_test_images_rgb = config.dir_test_images_rgb

        assert os.path.exists(self.dir_train_images_rgb)
        assert os.path.exists(self.dir_test_images_rgb)

    def visualize_img(self, img_idx: int = 1):
        grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.2)
        channel_names = ["yellow", "red", "blue"],
        descriptors = ["endoplasmic reticulum", "microtubules", "nucleus"]
        for i, (channel_name, description) in enumerate(zip(channel_names, descriptors)):
            if i == 2:
                ax = plt.subplot(grid[1, 0])
            else:
                ax = plt.subplot(grid[0, i])
            channel = os.path.join(self.dir_train_images_channels, f"{img_idx}_{channel_name}.png")
            read_channel = plt.imread(channel)
            ax.imshow(read_channel, aspect='auto', interpolation='nearest')
            ax.axis("off")
            plt.title(description)

        rgb_img = plt.imread(os.path.join(self.dir_train_images_rgb, f"{img_idx}.png"))
        plt.subplot(grid[1, 1]).imshow(rgb_img, aspect='auto', interpolation='nearest')
        plt.subplot(grid[1, 1]).axis("off")
        plt.title("RGB")
        plt.show()

    def save_rgb_images_train(self):
        for img_idx in self.range_train_imgs:
            blue, red, yellow = self.read_channels(self.dir_train_images_channels, img_idx)
            rgb_img = self.create_rgb(blue, red, yellow)
            cv2.imwrite(os.path.join(self.dir_train_images_rgb, f"{img_idx}.png"), rgb_img)

    def save_rgb_images_test(self):
        for img_idx in self.range_test_imgs:
            blue, red, yellow = self.read_channels(self.dir_test_images_channels, img_idx)
            rgb_img = self.create_rgb(blue, red, yellow)
            cv2.imwrite(os.path.join(self.dir_test_images_rgb, f"{img_idx}.png"), rgb_img)

    @staticmethod
    def read_channels(root, img_idx):
        filepath_blue_channel = os.path.join(root, f"{img_idx}_blue.png")
        filepath_red_channel = os.path.join(root, f"{img_idx}_red.png")
        filepath_yellow_channel = os.path.join(root, f"{img_idx}_yellow.png")

        assert all([os.path.exists(c) for c in (filepath_blue_channel, filepath_red_channel, filepath_yellow_channel)])

        blue = cv2.imread(filepath_blue_channel, 0)
        red = cv2.imread(filepath_red_channel, 0)
        yellow = cv2.imread(filepath_yellow_channel, 0)
        return blue, red, yellow

    @staticmethod
    def create_rgb(blue_channel_img, red_channel_img, yellow_channel_img):
        print("Remember blue channel shape:", blue_channel_img.shape)
        rgb_img = np.zeros((blue_channel_img.shape[0], blue_channel_img.shape[1], 3))
        rgb_img[:, :, 0] = blue_channel_img
        rgb_img[:, :, 1] = red_channel_img
        rgb_img[:, :, 2] = yellow_channel_img
        return rgb_img

    def train_data_with_labels(self):
        if os.path.exists(config.train_with_labels):
            with open(config.train_with_labels, "rb") as f:
                train_with_labels = np.load(f, allow_pickle=True)
                return train_with_labels

        imgs = []
        for img_idx in self.range_train_imgs:
            filepath = os.path.join(self.root_all_data, "rgb-train-images", f"{img_idx}.png")
            imgs.append(cv2.imread(filepath))
        y_train = pd.read_csv(os.path.join(f"{self.root_all_data}", "y_train.csv"))
        y_train.insert(1, "Images", imgs, True)
        with open(config.train_with_labels, "wb") as f:
            np.save(f, y_train)
        return y_train

    @staticmethod
    def train_and_validation_indices(X, y):
        def read_json():
            with open(config.train_and_test_indices) as json_file:
                data = json.load(json_file)
                train_indices = data["train"]
                test_indices = data["test"]
            return train_indices, test_indices

        if os.path.exists(config.train_and_test_indices):
            return read_json()

        sk_learn = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
        sk_learn.get_n_splits(X, y)
        train_and_text_indices_split = {}
        for train_indices, test_indices in sk_learn.split(X, y):
            train_and_text_indices_split["train"] = train_indices.tolist()
            train_and_text_indices_split["test"] = test_indices.tolist()
        with open(config.train_and_test_indices, 'w', encoding='utf-8') as outfile:
            json.dump(train_and_text_indices_split, outfile, ensure_ascii=False, indent=4)
        return read_json()


if __name__ == "__main__":
    cell_lines_preprocessing = CellLinesMicroscopyData()
    cell_lines_preprocessing.visualize_img(img_idx=60)
