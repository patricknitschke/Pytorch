import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

from tqdm import tqdm

# Device configuration
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(device)

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 400
latent_dim = 50 
num_classes = 10
num_epochs = 3
batch_size = 100
learning_rate = 1e-3

mnist_transform = transforms.Compose([
        transforms.ToTensor(),
])

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=mnist_transform,  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=mnist_transform)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

"""
    A simple implementation of Gaussian MLP Encoder and Decoder, from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial
"""
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parameters of simple tractable normal distribution "q"
        
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = th.sigmoid(self.FC_output(h))
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = th.randn_like(var).to(device)        # sampling epsilon        
        z = mean + var*epsilon                         # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, th.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

encoder = Encoder(input_dim=input_size, hidden_dim=hidden_size, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_size, output_dim = input_size)

model = Model(Encoder=encoder, Decoder=decoder).to(device)


# Define Loss function

BCE_loss = nn.BCELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * th.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

optimizer = Adam(model.parameters(), lr=learning_rate)

# Training 
print("Start training VAE...")
model.train()
for epoch in range(num_epochs):
    since = time.time()
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(tqdm(train_loader)):
        x = x.view(batch_size, input_size).to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    time_elapsed = time.time() - since
    print("\tEpoch", epoch + 1, "complete!", f"\tTime: {time_elapsed}", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    
print("Finish!!")

# Evaluation
model.eval()

examples = iter(test_loader)
example_batch1, _ = examples.next()

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_batch1[i][0], cmap='gray')
plt.show()

x_hat_batch = []
with th.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, input_size)
        x = x.to(device)
        
        x_hat, _, _ = model(x)
        x_hat_batch = (x_hat)
        break

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_hat_batch[i].view(28,28), cmap='gray')
plt.show()

