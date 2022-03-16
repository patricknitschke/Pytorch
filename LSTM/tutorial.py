import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 28 # 28 rows and 28 cols
sequence_length = 28
hidden_size = 128
num_layers = 2
batch_size = 100
num_epochs = 10


# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./MNIST_data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(
    root='./MNIST_data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes) -> None:
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # x shape: (batch_size, seq_len, input_size)

    def forward(self, x):
        pass

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(images.size(0), sequence_length, input_size).to(device)
