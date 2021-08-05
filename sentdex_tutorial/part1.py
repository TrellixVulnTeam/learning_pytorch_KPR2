import torch

# a tensor is essentially a multi-dimensional array
x = torch.tensor([5,3])
y = torch.tensor([2,1])
print(x * y)

# zeros
x = torch.zeros([2,5])
x.shape # gets the size

# rand
y = torch.rand([2,5])

# view (essentially a reshape)
y = y.view([1,10]) # reshaped from 2x5 to 1x10
print(y.shape)
 