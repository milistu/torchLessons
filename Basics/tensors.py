import torch
import numpy as np
### Torch documentation: https://pytorch.org/docs/stable/torch.html

print('-'*100)
print(r'>>>Create tensor<<<') 
print('Directly from data:')
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f'Data: \n {data}')
print(f'Tensor from data: \n {x_data}')

print('-'*100)
print('From numpy:')
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f'Numpy array: \n {np_array}')
print(f'Tensor from numpy: \n {x_np}')

print('-'*100)
print('From another tensor:')
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f'Ones Tensor: \n {x_ones}')
print(f'Random Tensor: \n {x_rand}')

print('-'*100)
print('With random or const values:')
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f'Random Tensor: \n {rand_tensor}')
print(f'Ones Tensor: \n {ones_tensor}')
print(f'Zeros Tensor: \n {zeros_tensor}')

print('-'*100)
print(r'>>>Tensor Atributes<<<')
# shape, datatype, stored device
tensor = torch.rand(3,4)

print(f'Shape of tensor: {tensor.shape}')
print(f'Datatype of tensor: {tensor.dtype}')
print(f'Device tensor is stored on: {tensor.device}') 

print('-'*100)
print(r'>>>Operations on Tensors<<<')

print('We move our tensor to GPU if availble')
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
print(f'Device tensor is stored on: {tensor.device}')

print('Standard numpy-like indexing and slicing:')
tensor = torch.ones(4, 4)
print(f'First row: {tensor[0]}')
print(f'First columns: {tensor[:, 0]}')
print(f'Last column: {tensor[..., -1]}')
tensor[:, 1] = 0
print(f'Tensor: \n {tensor}')

print('-'*100)
print('Joining tensors:')
t1 = torch.cat([tensor, tensor, tensor], dim=1)
t2 = torch.stack([tensor, tensor, tensor], dim=1)
print(f'Tensor concatenate: \n {t1}')
print(f'Tensor stacked: \n {t2}')

print('-'*100)
print('Arithmetic operations:')
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
print(f'y1 = \n {y1}')
y2 = tensor.matmul(tensor.T)
print(f'y2 = \n {y2}')
y3 = torch.rand_like(y1)
print(f'Before y3 = \n {y3}')
torch.matmul(tensor, tensor.T, out=y3)
print(f'After y3 = \n {y3}')

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
print(f'z1 = \n {z1}')
z2 = tensor.mul(tensor)
print(f'z2 = \n {z2}')

z3 = torch.rand_like(tensor)
print(f'Before z3 = \n {z3}')
torch.mul(tensor, tensor, out=z3)
print(f'After z3 = \n {z3}')

print('-'*100)
print('Single-element tensors: ')
# If you have a one-element tensor, 
# for example by aggregating all values of a tensor into one value, 
# you can convert it to a Python numerical value using item():
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print('-'*100)
print('In-place operations:')
# Operations that store the result into the operand are called in-place. 
# They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
print(f'{tensor} \n')
tensor.add_(5)
print(tensor)

# NOTE: In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. 
# Hence, their use is discouraged. 

print('-'*100)
print(r'>>>Bridge with NumPy<<<')
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
print('Tensor to NumPy array')
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')

t.add_(1)
print(f't: {t}')
print(f'n: {n}')

print('-'*100)
print('NumPy array to Tensor')
n = np.ones(5)
t = torch.from_numpy(n)
print(f't: {t}')
print(f'n: {n}')

np.add(n, 1, out=n)
print(f't: {t}')
print(f'n: {n}')

