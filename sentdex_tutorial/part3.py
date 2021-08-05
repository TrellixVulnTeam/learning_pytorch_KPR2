import torch
import torch.nn as nn
import torch.nn.functional as F

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
print(net)

X = torch.rand((28,28))

output = net(X.view(1,28*28))
print(output)