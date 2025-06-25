import torch
from torch.utils import data
from torch import nn

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)  # -1表示自动推导

true_w=torch.tensor([2, -3.4])
true_b=4.2
features, labels = synthetic_data(true_w, true_b, 1000)


def load_array(data_array, batch_size):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_array)  # *data_array参数解包（unpacking）,可索引，每次返回一个元组如(feature[i], label[i])
    return data.DataLoader(dataset, batch_size, shuffle=True)  # shuffle=True随机打乱数据

batch_size = 10
data_iter = load_array((features, labels), batch_size)  #(features, labels)组成元组

# 查看数据迭代器的第一个batch，需要显式调用 iter() 才能使用 next() 函数
# features_batch, labels_batch=next(iter(data_iter))  
# print(features_batch)
# print(labels_batch)

net=nn.Sequential(nn.Linear(2,1))  # nn.Sequential()是一个容器，可以将多个层组合在一起
print(net[0].weight.requires_grad)  # True，默认设置了requires_grad=True
print(net[0].bias.requires_grad)  # True

net[0].weight.data.normal_(0, 0.01)  # 正态分布初始化权重
net[0].bias.data.fill_(0)  # 初始化偏置为0

loss=nn.MSELoss()  # 均方误差损失函数

trainer=torch.optim.SGD(net.parameters(), lr=0.03)  # 随机梯度下降优化器，lr学习率

num_epochs=3  # 迭代次数
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 前向传播
        trainer.zero_grad()  # 梯度清零
        l.backward()  # 反向传播
        trainer.step()  # 更新参数
    l = loss(net(features), labels)  # 每个epoch结束后计算损失
    print(f'epoch {epoch + 1}, loss {l:f}')  # f-string格式化输出