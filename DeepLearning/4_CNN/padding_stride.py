import torch
from torch import nn

def comp_conv2d(conv2d, X):
    """Compute the output of a 2D convolutional layer."""
    X = X.reshape((1, 1) + X.shape)  # 添加批量和通道维度
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 去掉批量和通道维度

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)  
X = torch.rand(size=(8, 8))  
print(comp_conv2d(conv2d, X).shape) # (8, 8)

conv2d = nn.Conv2d(1, 1, kernel_size=(5,3), padding=(2,1))
print(comp_conv2d(conv2d, X).shape) # (8, 8)

conv2d = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
print(comp_conv2d(conv2d, X).shape) # (4, 4)

conv2d = nn.Conv2d(1, 1, kernel_size=(3,5), stride=(3,4), padding=(0,1))
print(comp_conv2d(conv2d, X).shape) # (2, 2)
