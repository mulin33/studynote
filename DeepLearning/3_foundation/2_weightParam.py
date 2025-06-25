import torch
from torch import nn

#----------------------------------------网络参数----------------------------------------
net = nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    nn.Linear(8,1)
)
X = torch.rand(2, 4)  
print(net(X))
print(net[2].state_dict())  # 查看第二层的参数
print(type(net[2].bias))    # torch.nn.parameter.Parameter
print(type(net[2].weight))   # torch.nn.parameter.Parameter
print(net[2].weight)  # tensor([-0.3390,...], requires_grad=True)
print(net[2].weight.data) # tensor([-0.3390,...])
print(net[2].weight.grad)  # None, 因为还没有反向传播

# -----查看所有参数---------
print(*[(name, param.shape) for name, param in net[0].named_parameters()], sep='\n')
print(*[(name, param.shape) for name, param in net.named_parameters()], sep='\n')
print(net.state_dict()['2.weight'].data)  

# -----嵌套块--------------
def block1():
    return nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)
print(rgnet(X)) 


# ------初始化------
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal)  # 迭代每个层进行初始化
print(net[2].weight.data)  

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42) # 42 宇宙的终极答案
        nn.init.zeros_(m.bias)
net.apply(init_constant)  
print(net[2].weight.data)  

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

net[0].apply(init_xavier)  # 只对第一层进行Xavier初始化
net[2].apply(init_constant)
print(net[0].weight.data)
print(net[2].weight.data)

# ------自定义初始化------
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *=m.weight.data.abs() >= 5  # 只保留绝对值大于等于 5 的权重，其它权重置为 0
net.apply(my_init)  
print(net[0].weight.data)  

# ------参数绑定(共享权重) ------
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.Linear(8, 1)
)
print(net[2].weight.data == net[4].weight.data)  # True, 共享权重