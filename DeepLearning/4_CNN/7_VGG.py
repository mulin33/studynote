import torch
from torch import nn
import sys
import os
sys.path.append(os.path.abspath("1_linearNet"))  
import softmaxClassify
import LeNet

def vgg_block(num_convs, in_channels, out_channels):
    """VGG block with multiple convolutional layers."""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))   # (num_convs, out_channels))
def vgg(conv_arch):
    """VGG11 model. 11=8conv + 3fc."""
    conv_blks = []
    in_channels = 1  # Input channels for grayscale images
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    conv_blks.append(nn.Flatten())
    conv_blks.append(nn.Linear(out_channels * 7 * 7, 4096))  # Assuming input size is 224x224
    conv_blks.append(nn.ReLU())
    conv_blks.append(nn.Dropout(0.5))
    conv_blks.append(nn.Linear(4096, 4096))
    conv_blks.append(nn.ReLU())
    conv_blks.append(nn.Dropout(0.5))
    conv_blks.append(nn.Linear(4096, 10))  # Output layer for 10 classes
    return nn.Sequential(*conv_blks)

net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))  
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]   # 把网络的宽度（通道数）缩小 4 倍,'//'是向下取整除法
small_net = vgg(small_conv_arch)

lr, num_epochs ,batch_size= 0.05, 10, 128
train_iter, test_iter = softmaxClassify.load_data_fashion_mnist(batch_size=batch_size, resize=224)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LeNet.train_ch6(small_net, train_iter, test_iter, num_epochs, lr, device)
softmaxClassify.stick_show()  