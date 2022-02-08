import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import optuna

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784 # 28x28
batch_size = 64
num_classes = 10
num_epochs = 2


# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./MNIST_data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(
    root='./MNIST_data', train=False, transform=transforms.ToTensor())

train_loader = th.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = th.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()

print(samples.shape, labels.shape) # torch.Size([64, 1, 28, 28]) torch.Size([64]) -- [batch, 1, 28, 28], [batch]
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

class NumberClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NumberClassifier, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # Don't use activation in last layer since we use CE loss (nn.LogSoftmax + nn.NLLLoss) (NegLogLikelihood Loss) 
        return out


# Loss and Optimiser
loss_func = nn.CrossEntropyLoss()

def optimise_model(trial):
    
    hidden_size = trial.suggest_categorical('hidden_size', [8,16,32,64])
    model = NumberClassifier(input_size, hidden_size, num_classes)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    n_total_steps = len(train_loader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 64, 1, 28, 28
            # 64, 784 -> reshape 
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = loss_func(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                print(f"epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")
            
    # Testing

    with th.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = th.max(outputs, 1) # returns value and index

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f"accuracy = {acc}")
    
    return acc

study = optuna.create_study(direction="maximize")
study.optimize(optimise_model, n_trials=50)

print(study.best_params) # E.g. {'x': 2.002108042}