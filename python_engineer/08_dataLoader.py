"""
epoch = 1 forward and backward pass of ALL training samples

batch_size = number of training samples in one forward & backward pass

number of iterations = number of passes, each pass using [batch_size]

"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
first_data = dataset[0]
features, label = first_data
# print('--- features & label for first index ---')
# print(features, label)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# training loop
NUM_EPOCHS = 2
TOTAL_SAMPLES = len(dataset)
N_ITERATIONS = math.ceil(TOTAL_SAMPLES / 4)

for epoch in range(NUM_EPOCHS):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{NUM_EPOCHS}, step {i+1}/{N_ITERATIONS}, inputs {inputs.shape}')
