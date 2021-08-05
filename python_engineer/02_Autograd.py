import torch

# calculate gradient w.r.t x
x = torch.randn(3, requires_grad=True)
print(x)



y = x * 2
print(f'this is a scalar value  :  {y}')
v = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
y.backward(v)
print(y)


# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():


x = torch.randn(3, requires_grad=True)
x.requires_grad_(False)
y = x.detach()

with torch.no_grad():
    y = x + 2


weights = torch.ones(4, requires_grad=True)

for epoch in range(5):
    model_output = (weights*3).sum()
    model_output.backward()

    # empty the gradients
    print(weights.grad.zero_())