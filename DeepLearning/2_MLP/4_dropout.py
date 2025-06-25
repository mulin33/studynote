import torch
from torch import nn

import sys
import os
sys.path.append(os.path.abspath("1_linearNet"))  
import softmaxClassify

def droupout_layer(X,dropout):
    """Dropout layer"""
    assert 0 <= dropout <= 1, "dropout must be between 0 and 1"
    if dropout == 0:
        return X
    if dropout == 1:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()  
    return mask * X / (1 - dropout)  # 这里做乘法比X[mask]速度更快

# X= torch.arange(16, dtype=torch.float32).reshape((2, 8))  
# print(X)
# print(droupout_layer(X, 0))  # dropout为0，返回原始数据
# print(droupout_layer(X, 0.5))  # dropout为0.5，返回随机mask后的数据
# print(droupout_layer(X, 1))  # dropout为1，返回全0数据

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(dropout1),  # 使用PyTorch内置的Dropout层
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout2),  # 使用PyTorch内置的Dropout层
    nn.Linear(num_hiddens2, num_outputs)
)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 0.01)  # 正态分布初始化权重
net.apply(init_weights) 

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')  

trainer_iter, test_iter = softmaxClassify.load_data_fashion_mnist(batch_size) 
trainer = torch.optim.SGD(net.parameters(), lr=lr)  
softmaxClassify.train_ch3(net, trainer_iter, test_iter, loss, num_epochs, trainer) 
softmaxClassify.stick_show()