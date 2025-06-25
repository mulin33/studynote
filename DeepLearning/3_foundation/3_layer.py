import torch
from torch import nn
from torch.nn import functional as F

# -------------------------自定义带参数的层-------------------------

class CenteredLayer(nn.Module):
    """自定义带参数的层"""
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return X - X.mean()  
    
# net=nn.Sequential(
#         nn.Linear(8, 128),
#         CenteredLayer())

class MyLinear(nn.Module):
    """自定义线性层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        # linear = X @ self.weight + self.bias    # @是torch.matmul的简写，梯度会自动传播
        linear = torch.matmul(X, self.weight.data) + self.bias.data  # 使用.data会避免梯度传播,具体比较可见3_layer_usedata.py
        return F.relu(linear)

net = nn.Sequential(
    MyLinear(64, 8),
    MyLinear(8, 1))



if __name__ == "__main__":
    
    # print(net(torch.rand(4, 8)))
    dense = MyLinear(5,3)
    print(dense.weight)
    print(dense(torch.rand(2, 5)))
    print(net(torch.rand(2, 64)))