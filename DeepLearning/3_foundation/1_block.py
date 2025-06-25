import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    """自定义MLP层"""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
class MySequential(nn.Module):
    """自定义顺序容器"""
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block]= block
    
    def forward(self, X):
        for block in self._modules:
            X = block(X)
        return X

class FixedHiddenMLP(nn.Module):
    """自定义固定隐藏层的MLP"""
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20,20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    
    
net1=nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

X = torch.rand(2, 20)  
print(net1(X))

net2 = MLP()  
print(net2(X))  

net3= MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(net3(X))