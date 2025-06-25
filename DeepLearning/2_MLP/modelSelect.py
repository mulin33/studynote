#----------------------为了其他py文件能导入该文件中的函数，2_modelSelect.py改名为modelSelect.py----------------------
import numpy as np
import math
import torch
from torch import nn
from torch.utils import data

import sys
import os
sys.path.append(os.path.abspath("1_linearNet"))  
import softmaxClassify

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上的损失"""
    metric = torch.zeros(1)
    for X, y in data_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        metric += l.sum()
    return metric.item() / len(data_iter.dataset)

def train(train_features,test_features,train_labels,test_labels,num_epochs):
    loss=nn.MSELoss()
    input_shape=train_features.shape[-1]    # 获得tarin_features的最后一个维度大小，即特征数量
    net=nn.Sequential(nn.Linear(input_shape, 1, bias=False))  
    batch_size=min(10, train_features.shape[0]) 
    train_iter=data.DataLoader(data.TensorDataset(train_features, train_labels.reshape(-1, 1)),batch_size, shuffle=True)
    test_iter=data.DataLoader(data.TensorDataset(test_features, test_labels.reshape(-1, 1)),batch_size, shuffle=False)
    trainer=torch.optim.SGD(net.parameters(), lr=0.01) 

    animator=softmaxClassify.Animator(xlabel='epoch', ylabel='loss',
                            legend=['train', 'test'], xlim=[1, num_epochs], ylim=[1e-4, 4])

    for epoch in range(num_epochs):
        softmaxClassify.train_epoch_ch3(net, train_iter, loss, trainer)  
        if epoch == 0 or (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, 
                       (evaluate_loss(net, train_iter, loss), 
                       evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy().flatten()) 


if __name__ == "__main__":

    max_degree=20
    n_train,n_test=100,100
    true_w=np.zeros(max_degree)
    true_w[0:4]=np.array([5,1.2,-3.4,5.6])  # 多项式真实系数
    print('true_weight:', true_w[0:4])

    features=np.random.normal(size=(n_train+n_test,1))  # 原始输入特征
    np.random.shuffle(features)
    poly_features=np.power(features, np.arange(max_degree).reshape(1,-1))   # 多项式扩展特征
    for i in range(max_degree):
        poly_features[:,i]/=math.gamma(i+1)

    labels=np.dot(poly_features,true_w)
    labels+=np.random.normal(scale=0.1,size=labels.shape)   # d2l中加了噪声的多项式模型

    # 将数据转换为torch张量
    true_w,features,poly_features,labels=[torch.tensor(x, dtype=torch.float32) 
                                        for x in [true_w, features, poly_features, labels]]
    # print(features[:2])
    # print([poly_features[:2,:]])
    # print(labels[:2])

    # 正常拟合
    # train(poly_features[:n_train,:4], poly_features[n_train:,:4], 
    #       labels[:n_train], labels[n_train:], num_epochs=50)

    # 欠拟合
    # train(poly_features[:n_train,:2], poly_features[n_train:,:2], 
    #       labels[:n_train], labels[n_train:], num_epochs=200)

    # 过拟合
    train(poly_features[:n_train,:], poly_features[n_train:,:], 
        labels[:n_train], labels[n_train:], num_epochs=50)
    
    softmaxClassify.stick_show()  # 显示图形