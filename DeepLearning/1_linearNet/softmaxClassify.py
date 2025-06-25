#----------------------为了其他py文件能导入该文件中的函数，文件名只能命名为softmaxClassify.py----------------------
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils import data

# trans=transforms.ToTensor()
# mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, download=False, transform=trans)    # 训练集,此时类似一个tensor的列表，通过索引获得tensor
# mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, download=False, transform=trans)
# print(len(mnist_train), len(mnist_test))  # 60000 10000

# # 查看数据集的第一个样本
# image,label=mnist_train[0]  
# print(type(image))  # <class 'torch.Tensor'>
# print(image.shape)  # torch.Size([1, 28, 28]),灰度图像 (对于tensor来说，通道数在前 C x H x W)
# print(label)
# plt.imshow(image.squeeze().numpy(), cmap='gray')  # squeeze()去掉维度为1的维度
# plt.title(f"{label}")  # label是tensor，item()将其转换为数字
# plt.axis('off')  # 关闭坐标轴
# plt.show()

# def get_fashion_mnist_labels(labels):   
#     """Return text labels for the Fashion-MNIST dataset."""
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]  # int()将tensor转换为数字

def get_dataloader_workers(): 
    """Use 8 processes to read the data."""
    return 8  # 返回进程数，num_workers=8,表示使用8个进程读取数据
batch_size = 256

# train_iter=data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())  # 256个样本一批，shuffle=True随机打乱数据

# timer=d2l.Timer()  # 计时器(加载数据的时间要远小于训练时间？)
# for X, y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec')  

#-------------------------softmax回归从0开始实现--------------------------------

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

# batch_size = 256
# num_inputs = 784  # 28*28
# num_outputs = 10  # 10个类别
# train_iter, test_iter = load_data_fashion_mnist(batch_size)  

# W=torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 权重初始化
# b=torch.zeros(num_outputs, requires_grad=True)  # 偏置初始化

# X=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # torch.Size([2, 3])
# print(X.sum(axis=0,keepdim=True).shape) # torch.Size([1, 3]),axis=0表示按行求和
# print(X.shape)
# print(X.sum(axis=1,keepdim=True))  # torch.Size([2, 1]),axis=1表示按列求和

def softmax(X):  
    """Compute the softmax for each row of the input X."""
    X_exp = X.exp()  # torch.Size([batch_size, num_outputs])
    partition = X_exp.sum(axis=1, keepdim=True)  # torch.Size([batch_size, 1])
    return X_exp / partition  # 应用了广播机制，torch.Size([batch_size, num_outputs]),每一行的和为1，归一化成概率分布

# X=torch.normal(0, 1, (2, 5))  # torch.Size([2, 5])
# X_prob = softmax(X)  # torch.Size([2, 5])
# print(X_prob)
# print(X_prob.sum(1))    #tensor([1., 1.])  # 每一行的和为1

def net(X):  # X是一个batch的样本,一个样本/图片是一行
    """The softmax regression model."""
    return softmax(torch.matmul(X.reshape((-1, num_inputs)), W) + b)  # torch.Size([batch_size, num_outputs]),reshape将X变为(batch_size, num_inputs)的形状

# y=torch.tensor([0, 2])  # 对第一个样本，取 y[0] = 0 类的概率；对第二个样本，取 y[1] = 2 类的概率
# y_hat=torch.tensor([[0.1, 0.2, 0.7], [0.1, 0.1, 0.8]])  
# print(y_hat[[0,1], y]) # pytorch的高级索引用法，取出y标签对应的概率

def cross_entropy(y_hat, y):  
    """Cross-entropy loss."""
    eps=1e-9  # 防止log(0)的情况
    return -torch.log(y_hat[range(len(y_hat)), y]+eps)  # torch.Size([batch_size]),取出y标签对应的概率，返回每个样本的损失值

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


class Animator:
    """在动画中绘制数据（适用于 VSCode 或普通 Python 脚本）"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(6, 4)):
        # 初始化图表
        if legend is None:
            legend = []
        plt.ion()  # 开启交互模式
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        self.fmts = fmts
        self.X, self.Y = None, None

        # 设置坐标轴
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )

    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """配置坐标轴属性"""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    def add(self, x, y):
        """添加数据点并更新图形"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        self.axes[0].cla()  # 清空原图
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.pause(0.001)  # 暂停以刷新图像

    def close(self):
        """手动关闭图形（可选）"""
        plt.ioff()
        plt.show()

def stick_show():
    """显示图形"""
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示图形，阻塞窗口直到关闭

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', ylabel='loss', legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # 训练一个epoch
        test_acc = evaluate_accuracy(test_iter, net)  # 测试集准确率
        animator.add(epoch + 1, [train_metrics[0], train_metrics[1], test_acc])  # 添加数据到动画
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:f}, train acc {train_metrics[1]:f}, test acc {test_acc:f}')
    plt.show()  # 显示动画图


def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():  
        for param in params:
            param -= lr * param.grad / batch_size  # 更新参数
            param.grad.zero_()  # 梯度清零
def updater(batch_size):
    """Return a function that updates parameters."""
    return sgd([W, b], lr, batch_size)  # 返回一个更新函数

if __name__ == '__main__':
    num_epochs = 10  
    lr=0.1
    batch_size = 256
    num_inputs = 784  # 28*28
    num_outputs = 10  # 10个类别

    train_iter, test_iter = load_data_fashion_mnist(batch_size)  

    W=torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 权重初始化
    b=torch.zeros(num_outputs, requires_grad=True)  # 偏置初始化

    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)  # 训练模型
    stick_show()  