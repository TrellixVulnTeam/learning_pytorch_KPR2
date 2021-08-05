import torch
import torch.utils.data
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

train = datasets.MNIST("", 
                       train=True, 
                       download=False, 
                       transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", 
                       train=False, 
                       download=False, 
                       transform=transforms.Compose([transforms.ToTensor()]))

"""
Batch Size -> how many data at a time you want to pass to your model
Common batch size 8 ~ 64 (base8 numbers)

Shuffle to get better generalization

"""
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

total = 0
counter = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0 }

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter[int(y)] += 1
        total += 1
print(counter)

for i in counter:
    print(f"{i}: {counter[i]/total*100}")