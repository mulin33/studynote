import torch
from torch import nn
import conv_oneChannel

def corr2d_multi_in(X, K):
    """2D cross-correlation for multiple input channels."""
    """多通道2d卷积核，而视频，医学图像是3d卷积核"""
    return sum(conv_oneChannel.corr2d(x, k) for x, k in zip(X, K))  # zip用法灵活，用于多通道

def corr2d_multi_in_out(X, K):
    """2D cross-correlation for multiple input and output channels."""
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)  # stack将结果堆叠成一个新的维度

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]], dtype=torch.float32)
print(K.shape)
print(corr2d_multi_in(X, K))  

K = torch.stack((K, K + 1, K + 2), dim=0)  
print(K.shape)
print(corr2d_multi_in_out(X, K))

def corr2d_multi_in_out_1x1(X, K):
    """2D cross-correlation for multiple input and output channels with 1x1 kernel."""
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape(c_i, h * w) 
    K = K.reshape(c_o, c_i)
    Y = torch.matmul(K, X)  # 全连接矩阵乘法 
    return Y.reshape(c_o, h, w)
 
X = torch.normal(0, 1, (3, 3, 3))  # 3个通道的3x3图像
K = torch.normal(0, 1, (2, 3, 1, 1))  # 2个输出通道，3个输入通道，1x1卷积核
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6, "The results are not equal!"