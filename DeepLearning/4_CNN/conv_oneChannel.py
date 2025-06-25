import torch
from torch import nn

def corr2d(x, k):
    """2D cross-correlation of x and k."""
    h,w = k.shape
    y = torch.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = (x[i:i+h, j:j+w] * k).sum()
    return y

class Conv2d(nn.Module):
    """2D convolutional layer."""
    def __init__(self, kernel_size):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
if __name__ == "__main__":
    x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.float32)
    k = torch.tensor([[0, 1], [2, 3]], dtype=torch.float32)
    y = corr2d(x, k)
    print(y)

    X1 = torch.ones((6, 8), dtype=torch.float32)
    X1[:, 2:6] = 0
    print(X1)
    K1 = torch.tensor([[1.0, -1.0]])
    Y1 = corr2d(X1, K1)
    print(Y1)

    # 学习一个卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False) 
    X1 = X1.reshape((1, 1, 6, 8))  # 添加批量和通道维度
    Y1 = Y1.reshape((1, 1, 6, 7))  

    for i in range(20):
        Y1_hat = conv2d(X1)
        l = (Y1_hat - Y1).pow(2).sum()
        conv2d.zero_grad()
        l.backward()
        conv2d.weight.data -= 0.01 * conv2d.weight.grad
        print(f'epoch {i + 1}, loss {l.item():.3f}')

    print(f'weight: {conv2d.weight.data.reshape((1, 2))}')