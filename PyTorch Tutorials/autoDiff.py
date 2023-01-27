import torch

### AUTOMATIC DIFFERENTIATION WITH TORCH.AUTOGRAD

x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad=True) # or later: w.requires_grad_(True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

## A reference to the backward propagation function is stored in grad_fn property of a tensor. 
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

print('-'*100)
## Computing Gradients
loss.backward()
print(f'Gradient of W: {w.grad}')
print(f'Gradient of B: {b.grad}')

print('-'*100)
## Disabling Gradient Tracking
# Exp: When computing forward propagation
'''
    There are reasons you might want to disable gradient tracking:
    --------------------------------------------------------------
        To mark some parameters in your neural network as frozen parameters. 
            This is a very common scenario for finetuning a pretrained network

        To speed up computations when you are only doing forward pass, 
            because computations on tensors that do not track gradients would be more efficient.

'''

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x , w) + b
print(z.requires_grad)

print('-'*50)
# Another way:
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

print('-'*100)
## Tensor Gradients and Jacobian Products

inp = torch.eye(4, 5, requires_grad=True)
# print(inp)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"Second call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradient\n{inp.grad}")








