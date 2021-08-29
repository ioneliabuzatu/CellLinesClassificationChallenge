import os

import experiment_buddy
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

torchvision_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "alexnet": models.alexnet,
    "vgg19": models.vgg19,
    "squeezenet": models.squeezenet1_0,
    "densenet": models.densenet161,
    "inception": models.inception_v3,
    "googlenet": models.googlenet,
    "shufflenet": models.shufflenet_v2_x1_0,
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0,
}

# weather to use buddy or not and run local or cluster
USE_BUDDY = True
LOCAL = True

# hyper-parameters
seed = 1234
batch_size = 64
lr = 0.0002
weight_decay = 0.0005
betas = (0.99, 0.95)
epochs = 25
which_resnet = "resnet50"

# paths
if LOCAL:
    root_all_data = "/home/ionelia/pycharm-projects/master/semester2/lifescience/summer/data/celllinesproject"
    save_model_file_path = f"./models/checkpoints/cell_lines_{which_resnet}_model.pth"
    dir_train_images_separate_channels = os.path.join(root_all_data, "images_train")
    dir_test_images_separate_channels = os.path.join(root_all_data, "images_test")
    dir_train_images_rgb = os.path.join(root_all_data, "rgb-train-images")
    dir_test_images_rgb = os.path.join(root_all_data, "rgb-test-images")
    train_with_labels = "train_images_with_labels.npy"
    train_and_test_indices = "train_and_test_indices.json"
else:
    root_all_data = "/home/mila/g/golemofl/data/cell"
    save_model_file_path = os.path.join(root_all_data, "cell_lines_model.pth")
    dir_train_images_separate_channels = os.path.join(root_all_data, "images_train")
    dir_test_images_separate_channels = os.path.join(root_all_data, "images_test")
    dir_train_images_rgb = os.path.join(root_all_data, "rgb-train-images")
    dir_test_images_rgb = os.path.join(root_all_data, "rgb-test-images")
    train_with_labels = "train_images_with_labels.npy"
    train_and_test_indices = "train_and_test_indices.json"

encode = {'A549': 0, 'CACO-2': 1, 'HEK 293': 2, 'HeLa': 3, 'MCF7': 4, 'PC-3': 5, 'RT4': 6, 'U-2 OS': 7, 'U-251 MG': 8}
decode = {v: k for k, v in encode.items()}


if USE_BUDDY:
    experiment_buddy.register_defaults(locals())
    tensorboard = experiment_buddy.deploy(
        "",
        sweep_yaml="",
        proc_num=1,
        wandb_kwargs={"project": "celllinesclassifier"},
    )
else:
    tensorboard = SummaryWriter()
