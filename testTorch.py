import torch

print('-'*100)
print(f'Is torch available: {torch.cuda.is_available()}')
print('-'*100)
print(f'Current device: {torch.cuda.current_device()}')
print('-'*100)
print(f'Device with index 0: {torch.cuda.device(0)}')
print('-'*100)
print(f'Number of devices on machine: {torch.cuda.device_count()}')
print('-'*100)
print(f'Get device name on index 0: {torch.cuda.get_device_name(0)}')
print('-'*100)
