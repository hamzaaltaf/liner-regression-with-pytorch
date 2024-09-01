import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import math
import pandas as pd


class AmazonDataLoader(Dataset):
    def __init__(self, dataset):
        # data = np.loadtxt('./datasets/housing.csv', skiprows=1, delimiter=',')
        data = dataset.to_numpy()
        self.x = torch.from_numpy(data[:, 0:7].astype(np.float32))
        self.y = torch.from_numpy(data[:, 8].astype(np.float32))
        self.sampels = data.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.sampels
    

df = pd.read_csv('./datasets/housing.csv')
copy = df.copy()
copy = copy.drop(['ocean_proximity'], axis=1)

dataset = AmazonDataLoader(copy)
features, target = dataset[0]
print(features)
print(target)
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=4)
dataiter = iter(dataloader)
# hyper parameters
num_epochs = 100
total_sampels = len(dataset)
iterations = math.ceil((total_sampels/4))

for epoch in range(num_epochs):
    for i, (f, t) in enumerate(dataloader):
        print(f'epoch {epoch} with feature {f} and target {t}')