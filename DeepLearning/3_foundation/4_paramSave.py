import torch
from torch import nn
from torch.nn import functional as F

#----------tensor,list和dict的保存和加载----------
x=torch.arange(4)
torch.save(x,'x-save')

x2=torch.load('x-save',weights_only=True)  
print(x2)


y = torch.zeros(4)
torch.save([x, y], 'xy-save')  # 保存为列表
x2,y2 = torch.load('xy-save',weights_only=True)  
print(x2, y2)

mydict= {'x': x, 'y': y}
torch.save(mydict, 'mydict-save')  # 保存为字典
mydict2 = torch.load('mydict-save',weights_only=True)
# print(mydict2['x'], mydict2['y'])
print(mydict2)

#----------模型参数的保存和加载----------
class MLP(nn.Module):
    """自定义MLP层"""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
net=MLP()
X=torch.randn(2, 20)
Y=net(X)
print(Y)

torch.save(net.state_dict(), 'mlp-params')  # 保存模型参数

net2 = MLP()  # 创建一个新的模型实例
net2.load_state_dict(torch.load('mlp-params',weights_only=True))  # 加载参数
print(net2.eval())