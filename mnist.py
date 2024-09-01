import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#hyperparameters
input_size = 784
hidden_layers = 100
batch_size = 100
num_classes = 10
epochs = 2

train_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms.ToTensor(), train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', transform=transforms.ToTensor(), train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

example = iter(train_loader)
features, target = next(example)
print(features.shape, target.shape)

for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(features[i][0], cmap='grey')
plt.show()

# Nueral Net
class NueralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NueralNet, self).__init__()
        self.l1 =  nn.Linear(input_size, hidden_size)
        self.relu =  nn.ReLU()
        self.l2 =  nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    

model = NueralNet(input_size, hidden_layers, num_classes)

#loss
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# training loop
n_total_sets = len(train_loader)
for epoch in range(epochs):
    for i, (images, label) in iter(train_loader):
        images = images.reshape(-1, 28*28)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# evaluation
with torch.no_grad():
    n_correct = 0
    n_sample = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predictions =  torch.max(outputs, 1)
        n_correct += (predictions == labels).sum().item()
        n_sample += labels.shape[0]

    acc = 100* n_correct / n_sample
    print(acc)
