import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from di_dataset import DepthImageDataset

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/vae_cifar10')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Hyperparams
input_size = 270*480
hidden_dim = 16
latent_dim = 400 
num_epochs = 3
batch_size = 64
learning_rate = 1e-3

# Save path
saves_folders = "../../rl_data"
load_paths = [os.path.join(saves_folders, saves_folder) for saves_folder in os.listdir(saves_folders)]

# Obtain train and test data
dataset = DepthImageDataset(load_paths=load_paths)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0)

