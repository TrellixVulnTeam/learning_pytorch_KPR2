# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
from torch.nn.modules.loss import CrossEntropyLoss
import torch.utils
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28 x 28 image
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           transform=transforms.ToTensor(), download=False)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print('--sample data--')
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# plt.show()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):
        # reshape 100, 1, 28, 28 to 100, 784
        image = image.reshape(-1, 28*28).to(device)
        label = label.to(device)

        # forward pass
        outputs = model(image)
        loss = criterion(outputs, label)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for image, label in test_loader:
        image = image.reshape(-1, 28*28).to(device)
        label = label.to(device)
        # trained model now
        outputs = model(image)

        # value, index
        _, predictions = torch.max(outputs.data, 1)
        n_samples += label.size(0)
        n_correct += (predictions == label).sum().item()
    
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')