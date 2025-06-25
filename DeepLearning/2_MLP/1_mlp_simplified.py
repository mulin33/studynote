import torch
from torch import nn
import sys
import os
sys.path.append(os.path.abspath("1_linearNet"))  
import softmaxClassify


num_imputs,num_outputs,num_hiddens=784,10,256  # 输入层784，输出层10，隐藏层256

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_imputs, num_hiddens),
                    nn.ReLU(),
                    nn.Linear(num_hiddens, num_outputs))

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,0,0.01)
net.apply(init_weights)

batch_size,lr,num_epochs = 256,0.1,10
loss = nn.CrossEntropyLoss(reduction='none')  # 交叉熵损失函数，为什么这里不求平均？
trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 随机梯度下降优化器
train_iter, test_iter = softmaxClassify.load_data_fashion_mnist(batch_size)

softmaxClassify.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) 

softmaxClassify.stick_show()  # 显示图形