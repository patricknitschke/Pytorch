import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataSet(Dataset):
    def __init__(self) -> None:
        super(WineDataSet, self).__init__()
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataSet()

# first_row = dataset[0]
# features, labels = first_row
# print(features, labels)

batch_size = 4
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# dataiter = iter(dataloader)
# features, labels = dataiter.next()
# print(features)
# print(labels)

#training loop 
num_epochs = 2
total_samples = len(dataset)
total_iterations = math.ceil(total_samples/batch_size)
print(total_samples, total_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        
        # We get a list of batch size length, each element has length len(features), batch size labels
        # print(inputs, labels) 
        if (i+1) % 5 == 0:
            print(f"epoch: {epoch+1}/{num_epochs}, step: {i+1}/{total_iterations}, inputs: {inputs.shape}")