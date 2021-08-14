"""
based on
https://jovian.ai/aakashns/05-cifar10-cnn
"""

import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import torch.utils.data

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.facecolor'] = '#ffffff' 

from model import Cifar10CnnModel


##### Download & Extract dataset ######
PROJECT_NAME = '05-cifar10-cnn'
DOWNLOAD = False
EXTRACT = False

DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
if DOWNLOAD:
    download_url(DATASET_URL, '.')
else:
    print('Dataset downloaded.')

if EXTRACT:
    # Extract from archive
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')
else:
    print('Dataset folder extracted.')



##### PyTorch Image Folder class for preprocessing #####
print('---------'*10)

DATA_DIR = './data/cifar10'

"""
The dataset directory structure (one folder per class) is used by many computer vision datasets, 
and most deep learning libraries provide utilites for working with such datasets. 
We can use the ImageFolder class from torchvision to load the data as PyTorch tensors.
"""

dataset = ImageFolder(DATA_DIR+'/train', transform=ToTensor())
img, label = dataset[0]
print(f'Image Shape: {img.shape}, Image Label: {label}')
print(f'Label Classes: {dataset.classes}')



###### Show Example with matplotlib #####
def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
# show_example(*dataset[0])



##### Data PreProcessing (Training & Validation Split) #####
print('---------'*10)
random_seed = 42
torch.manual_seed(random_seed)

VAL_SIZE = 50
TRAIN_SIZE = len(dataset) - VAL_SIZE
BATCH_SIZE = 128

train_ds, val_ds = random_split(dataset, [TRAIN_SIZE, VAL_SIZE])
print(f'Length of Training Dataset {len(train_ds)}')
print(f'Length of Validation Dataset {len(val_ds)}')


train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


##### Calling Model #####
# print('---------'*10)
model = Cifar10CnnModel()
# print(model)


##### To check if GPU is vailable #####
print('---------'*10)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
print(f'Your current device is: {device}.')


train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)



##### Training the Model #####
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for step, batch in enumerate(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch: {epoch+1}/{epochs} Step: {step}/{len(train_loader)}, loss: {loss:.4f}')

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history


model = to_device(model, device)

NUM_EPOCHs = 5
OPT_FUNC = torch.optim.Adam
LEARNING_RATE = 0.001

print('---------'*10)
print('Starting Training....')
# history = fit(NUM_EPOCHs, LEARNING_RATE, model, train_dl, val_dl, OPT_FUNC)



##### Plots #####
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs No.of epochs')
    plt.show()


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()



#### Testing Trained Model #####
test_dataset = ImageFolder(DATA_DIR + '/test', transform=ToTensor())

def predict_image(img ,model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get Predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]


# Test 1
img, label = test_dataset[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

# Test 2
img, label = test_dataset[1002]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

# Test 3
img, label = test_dataset[6153]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

test_loader = DeviceDataLoader(DataLoader(test_dataset, BATCH_SIZE*2), device)
result = evaluate(model, test_loader)
print(result)


##### Saving & Loading the model #####
torch.save(model.state_dict(), 'cifar10-cnn.pth')

model2 = to_device(Cifar10CnnModel(), device)
model2.load_state_dict(torch.load('cifar10-cnn.pth'))

print(evaluate(model2, test_loader))

