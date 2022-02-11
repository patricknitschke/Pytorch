from pickletools import optimize
from statistics import mode
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyper-parameters 
input_size = 32*32 # 32x32
hidden_dim = 400
latent_dim = 50 
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 1e-3

cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 Data: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=cifar_transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                                download=True, transform=cifar_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


"""
    A simple implementation of Convolutional VAE, from https://github.com/ttchengab/VAE/blob/main/VAE.py 
"""
class Encoder(nn.Module):
    def __init__(self, input_channels, feature_dims, latent_dim=latent_dim) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3)
        self.f_mu = nn.Linear(hidden_dim*feature_dims, latent_dim)
        self.f_logvar = nn.Linear(hidden_dim*feature_dims, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        print(x.shape)

class VAE(nn.Module):
    def __init__(self) -> None:
        super(VAE, self).__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    model = Encoder(3,10)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(train_loader)

