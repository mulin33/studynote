import torch
from torch import nn

print(torch.cuda.device_count())  # 查看GPU数量
x1 = torch.tensor([1, 2, 3])
print(x1.device)
x2 = torch.ones((2, 3), device='cuda')  # 在GPU上创建张量
print(x2.device)