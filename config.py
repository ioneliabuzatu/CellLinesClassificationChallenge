import os

import experiment_buddy
from torch.utils.tensorboard import SummaryWriter

USE_BUDDY = True
LOCAL = True

if LOCAL:
    root_all_data = "/home/ionelia/pycharm-projects/master/semester2/lifescience/summer/data/celllinesproject"
    dir_train_images_separate_channels = os.path.join(root_all_data, "images_train")
    dir_test_images_separate_channels = os.path.join(root_all_data, "images_test")
    dir_train_images_rgb = os.path.join(root_all_data, "rgb-train-images")
    dir_test_images_rgb = os.path.join(root_all_data, "rgb-test-images")
    train_with_labels = "train_images_with_labels.npy"
    train_and_test_indices = "train_and_test_indices.json"
else:
    raise NotImplementedError

encode = {'A549': 0, 'CACO-2': 1, 'HEK 293': 2, 'HeLa': 3, 'MCF7': 4, 'PC-3': 5, 'RT4': 6, 'U-2 OS': 7, 'U-251 MG': 8}
decode = {v: k for k, v in encode.items()}

# hyper-parameters
seed = 1234
batch_size = 64
lr = 0.0002
weight_decay = 0.0005
betas = (0.99, 0.95)
epochs = 25

if USE_BUDDY:
    experiment_buddy.register(locals())
    tensorboard = experiment_buddy.deploy(
        "",
        sweep_yaml="",
        proc_num=1,
        wandb_kwargs={"entity": "CellLinesClassificationProject"},
    )
else:
    tensorboard = SummaryWriter()
