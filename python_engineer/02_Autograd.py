import torch

# calculate gradient w.r.t x
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2 
print(y)
z = y * y * 2
z = z.mean()
print(z)

z.backward() # dz/dx
print(x.grad)



# x.requires_grad_(False)
print('\n')
print('----- requires_grad_(False) example ------- ')
x = torch.randn(3, requires_grad=True)
print(x)
x.requires_grad_(False)
print(x)
print('\n')

# x.detach()
print('\n')
print('----- .detach() example ------- ')
x = torch.randn(3, requires_grad=True)
print(x)
print(x.detach())
print('\n')


# with torch.no_grad():
print('\n')
print('----- with torch.no_grad() example ------- ')
x = torch.randn(3, requires_grad=True)
print(x)
with torch.no_grad():
    y = x + 2
    print(y)
print('\n')


# example
print('\n')
print('----- weights example -----')
weights = torch.ones(4, requires_grad=True)

for epoch in range(5):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

    # empty the gradients (un-comment below to see difference)
    weights.grad.zero_()

"""
1. whenever calculating gradients, requires_grad=True
2. before doing the next iteration in optimization steps, must empty gradients; weights.grad_zero()
3. three different ways to detach gradient.
"""