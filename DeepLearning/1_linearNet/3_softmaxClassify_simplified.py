import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
import torch

def get_dataloader_workers(): 
    """Use 8 processes to read the data."""
    return 8  # 返回进程数，num_workers=8,表示使用8个进程读取数据

def load_data_fashion_mnist(batch_size, resize=None):  
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans=[transforms.ToTensor()]  # 定义为list,方便在有resize时insert
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, download=False, transform=trans)    # 训练集,此时类似一个tensor的列表，通过索引获得tensor
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, download=False, transform=trans)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))  # 返回训练集和测试集的迭代器

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

net=nn.Sequential(
    nn.Flatten(),  # 将输入的图片展平为一维向量，但保留batch_size维度
    nn.Linear(784, 10)  # 线性层，输入维度为784，输出维度为10
)

def init_weights(m):
    """Initialize weights."""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 正态分布初始化权重
        nn.init.zeros_(m.bias)  # 偏置初始化为0

net.apply(init_weights)  # 应用初始化函数到网络的每一层

loss=nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于多分类问题（里面已包含softmax实现）
trainer=torch.optim.SGD(net.parameters(), lr=0.1)  # 随机梯度下降优化器，lr学习率

class Accumulator:
    """自定义累加器，用于累加多个变量的总和"""
    def __init__(self, n):
        self.data = [0.0] * n  # 初始化 n 个累加器

    def add(self, *args):
        # 对传入的每个值，累加到对应位置，注意下zip()用法
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):  
    """Compute the accuracy."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # y_hat:batch个样本的概率分布
        y_hat = y_hat.argmax(axis=1)  # torch.Size([batch_size]),取出每一行的最大值的索引
    cmp = y_hat.type(y.dtype) == y  # 将y_hat转换为与y相同的类型，再判断是否和y相等
    return float(cmp.type(y.dtype).sum())  # 返回正确分类的总样本数

def evaluate_accuracy(data_iter, net):
    """Evaluate the accuracy of a model on the given data set."""
    # 在本文件中，net是自定义函数实现，为False
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式，这样可以关闭dropout等训练时特有的操作
    metric = Accumulator(2)  # 正确分类的样本数和总样本数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # numel()返回元素个数
    return metric[0] / metric[1]  # 返回正确分类的样本数和总样本数的比值
 
def train_epoch_ch3(net, train_iter, loss, updater):
    """Train a model for one epoch (defined in Chapter 3)."""
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式，这样可以启用dropout等训练时特有的操作
    metric = Accumulator(3)  # 累加器，记录损失，正确分类的样本数，总样本数
    for X, y in train_iter:
        y_hat = net(X)  # 前向传播
        l = loss(y_hat, y)  # 交叉熵损失函数
        if isinstance(updater, torch.optim.Optimizer):  
            updater.zero_grad()  
            l.mean().backward()  
            updater.step()  
        else:  # 如果updater是自定义的更新函数
            l.sum().backward()  
            updater(X.shape[0])  # 更新参数，传入batch_size
        metric.add(l.sum(), accuracy(y_hat, y), y.numel())  # 累加损失、正确分类的样本数和总样本数
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均损失和准确率

num_epochs = 10 
for epoch in range(num_epochs):
    train_metrics = train_epoch_ch3(net, train_iter, loss, trainer)  # 训练一个epoch
    test_acc = evaluate_accuracy(test_iter, net)  # 测试集准确率
    print(f'epoch {epoch + 1}, loss {train_metrics[0]:f}, train acc {train_metrics[1]:f}, test acc {test_acc:f}')  # 输出每个epoch的损失和准确率