import torch
from torch import nn
import sys
import os
sys.path.append(os.path.abspath("1_linearNet"))  
import softmaxClassify
import LeNet

net = nn.Sequential(
    # 使用一个11*11的卷积核，输出通道数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里全连接层的输出数量是LeNet的好几倍，使用dropout层来减轻过拟合
    nn.Linear(256 * 5 * 5, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

X = torch.randn(1, 1, 224, 224)  
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

batch_size = 128
train_iter, test_iter = softmaxClassify.load_data_fashion_mnist(batch_size=batch_size, resize=224)
lr, num_epochs = 0.01, 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LeNet.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
softmaxClassify.stick_show()  