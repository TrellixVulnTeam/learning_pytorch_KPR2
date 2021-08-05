import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__() # initializing from nn.module, inherit the init frm the parent class
        """
        fc -> fully connected layer
        28*28 = 784
        nn.Linear(input, output)

        10 at the final output cause 0~9 

        """
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3 # 3 whole passes through our entire dataset

for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward() # back propagation
        optimizer.step() # adjust the weights
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i  in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))