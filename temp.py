import torch
print(torch.cuda.is_available())       # Should be True
print(torch.cuda.current_device())     # Index of the GPU
print(torch.cuda.get_device_name(0))   # Name of your GPU
