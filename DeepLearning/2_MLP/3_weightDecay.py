from torch import nn
import torch

import sys
import os
sys.path.append(os.path.abspath("1_linearNet"))  
import linearRegress
import softmaxClassify
import modelSelect


n_train,n_test,num_inputs,batch_size=20,100,200,5
true_w,true_b=torch.ones(num_inputs,1)*0.01,0.05
trian_data=linearRegress.synthetic_data(true_w, true_b, n_train)  
train_iter=linearRegress.load_array(trian_data, batch_size)  
test_data=linearRegress.synthetic_data(true_w, true_b, n_test)
test_iter=linearRegress.load_array(test_data, batch_size, is_shuffle=False)

def init_params():
    """Initialize model parameters."""
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)  
    b = torch.zeros(1, requires_grad=True)  
    return [w, b]

def l2_penalty(w):
    """L2 norm penalty."""
    return torch.sum(w.pow(2)) / 2  

def train(lambd):
    """Train the model with weight decay."""
    w,b= init_params()  
    net,loss=lambda X: linearRegress.linreg(X, w, b), linearRegress.squared_loss
    num_epochs,lr=100,0.003
    animator=softmaxClassify.Animator(xlabel='epochs', ylabel='loss', legend=['train loss', 'test loss'],xlim=[5,num_epochs],yscale='log')

    for epoch in range(num_epochs):
        for X, y in train_iter:
            l=loss(net(X), y) + lambd * l2_penalty(w)  # 添加L2正则化项,注意，这里通过广播机制，l2_penalty(w)会自动扩展为与l相同的形状
            l.sum().backward()
            linearRegress.sgd([w, b], lr, batch_size)   # 在sgd中，会sum_loss/batch_size来更新参数
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (modelSelect.evaluate_loss(net, train_iter, loss),
                                   modelSelect.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数:', torch.norm(w).item())  

def train_concise(wd):
    """Train the model using concise API."""
    net = nn.Sequential(nn.Linear(num_inputs, 1))  
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003

    # trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)  # 这种方式会对weight和bias都添加L2正则化
    trainer = torch.optim.SGD([
        {'params':net[0].weight,'weight_decay':wd},  # 对权重添加L2正则化，而对偏置不添加L2正则化
        {'params':net[0].bias}],lr=lr)  # 在实践中，只对权重加正则，不对偏置和 BatchNorm 参数加正则
    
    animator = softmaxClassify.Animator(xlabel='epochs', ylabel='loss', legend=['train loss', 'test loss'], xlim=[5, num_epochs], yscale='log')

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l=loss(net(X),y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (modelSelect.evaluate_loss(net, train_iter, loss),
                                   modelSelect.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数:', net[0].weight.norm().item())  

# Train without weight decay
# train(lambd=0)

# Train with weight decay
# train(lambd=3)

# Train using concise API without weight decay
# train_concise(wd=0)  

# Train using concise API with weight decay
train_concise(wd=3)

softmaxClassify.stick_show()  # 显示图形

# note:实际发现w的L2范数也可以通过初始化时给一个更小的标准差来实现，但是weight_decay可以实现,w的L2范数更小，test loss更小，且训练更快