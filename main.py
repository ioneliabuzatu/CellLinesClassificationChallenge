import os

import numpy as np
from matplotlib import pyplot as plt


class CellLinesMicroscopyData:
    def __init__(self, root_dir_images="../data/celllinesproject/images_train/"):
        self.range_imgs = range(1, 9632)
        self.dir_train_images = root_dir_images

    def visualize_img(self, img_idx=1):
        grid = plt.GridSpec(1, 4, wspace=0.05, hspace=0.1)
        sample_dcit = {"size":(64,64)}
        channels = ["yellow", "red", "blue", ]
        for i, c in enumerate(["yellow", "red", "blue"]):
            ax = plt.subplot(grid[i])
            channel = os.path.join(self.dir_train_images, f"{img_idx}_{c}.png")
            read_channel = plt.imread(channel)
            # plt.imshow(read_yellow_channel)
            # ax.imshow(select_channels(sample_dict, [c, ]), aspect='auto', interpolation='nearest')
            ax.imshow(read_channel, aspect='auto', interpolation='nearest')
            # plt.title(labels[i], fontsize=18)
            ax.axis("off")
            sample_dcit[c] = read_channel

        ax = plt.subplot(grid[3])
        merged_channels = self.select_channels(sample_dcit, channels)
        ax.imshow(merged_channels, aspect='auto', interpolation='nearest')
        plt.show()

    def select_channels(self, sample_dict, channel_list):
        image_height, image_width = sample_dict["size"]
        zero_img = np.zeros([image_height, image_width], dtype=np.uint8)
        img = [
            # R
            sample_dict["red"] if "red" in channel_list else zero_img,
            # G
            sample_dict["green"] if "green" in channel_list else zero_img,
            # B
            sample_dict["blue"] if "blue" in channel_list else zero_img
        ]
        img = np.stack(img, axis=-1)
        if "yellow" in channel_list:
            yellow = sample_dict["yellow"]
            # overwrite first two channels for yellow -> rgb(255,255,0)
            img[:, :, :2] = np.stack([yellow, yellow], axis=-1)
        # overwrite with protein so we always see it
        if "green" in channel_list:
            img[:, :, 1] = sample_dict["green"]
        return img

    def __len__(self):
        return len(self.range_imgs)

    def __getitem__(self, idx):
        pass


cell_lines_dataset = CellLinesMicroscopyData()
cell_lines_dataset.visualize_img()