import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'torch version: {torch.__version__}')
print(f'device: {device}')
print(f'{torch.cuda.device_count()} GPU(s) available')
