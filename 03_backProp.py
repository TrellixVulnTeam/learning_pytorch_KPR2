# 1. Forward Pass: Compute Loss
# 2. Compute local gradients
# 3. Backward pass: Computer dLoss / dWeights using the Chain Rule

import torch

x = torch.tensor(1.0)
y  = torch.tensor(2.0)

# interested in the gradient for weights
w = torch.tensor(1.0, requires_grad=True)

# foward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)


# backward pass
loss.backward()
print(w.grad)

# update weights
# next forward and backward