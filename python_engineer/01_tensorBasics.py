import torch 

# Tensor can be 1D, 2D, 3D , 4D

x = torch.empty(2,3) # 2 x 3 matrix 
x = torch.rand(2,2) # 2 x 2 random matrix
x = torch.zeros(2,2) 
x = torch.ones(2,2, dtype=torch.int) # 2x2 ones matrix of type integer
x= torch.tensor([2, 6])
print(x.size())
print(x)

x = torch.rand(2,2)
y = torch.rand(2,2)
y.add_(x) # any underscore will change value of y
print(y)

# slicing 
x = torch.rand(5,3)
print(x[:, 1]) # all the rows and 2nd column
print(x[1,1].item()) # get the single item value

# Reshaping tensor, make sure the matrix adds up
x = torch.rand(4,4)
print(x)
y = x.view(16)
y = x.view(-1, 8) # automatically resize accordingly to ** by 8
print(y)

# if GPU available
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)

# to calculate gradient
x = torch.ones(5, requires_grad=True)