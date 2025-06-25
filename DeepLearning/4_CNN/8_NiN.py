import torch
from torch import nn
import sys
import os
sys.path.append(os.path.abspath("1_linearNet"))  
import softmaxClassify
import LeNet

def nin_block(in_channels, out_channels, kernel_size, strides=1, padding=0):
    """NIN block with 1x1 convolution."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),    # 1x1 convolution
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()   # 1x1 convolution
    )

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4), 
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, padding=2), 
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, padding=1), 
    nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, padding=1),   #[1, 10, 5, 5]
    nn.AdaptiveAvgPool2d((1, 1)),   # [1, 10, 1, 1]
    nn.Flatten()
)

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = softmaxClassify.load_data_fashion_mnist(batch_size=batch_size, resize=224)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LeNet.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
softmaxClassify.stick_show()