#----------------------为了其他py文件能导入该文件中的函数，文件名只能命名为softmaxClassify.py----------------------
import random
import torch
from matplotlib import pyplot as plt
from torch.utils import data

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)  # -1表示自动推导

def data_iter(X, y, batch_size):
    """Generate minibatches."""
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 随机打乱,只能作用于list
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i:min(i + batch_size, num_examples)])  # min防止越界
        # yield,类似return 但会暂停函数运行，把当前这批数据“交出去”，下次迭代再接着从这里执行，适合分批次生成数据，节省内存
        yield X.index_select(0, j), y.index_select(0, j)  # index_select选择行

def load_array(data_array, batch_size, is_shuffle=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_array)  # *data_array参数解包（unpacking）,可索引，每次返回一个元组如(feature[i], label[i])
    return data.DataLoader(dataset, batch_size, shuffle=is_shuffle)  # shuffle=True随机打乱数据

# 绘图显示数据
# plt.scatter(features[:, 0].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()

# 只打印第一批数据
# for X, y in data_iter(features, labels, bathch_size):
#     print(X, '\n', y)
#     break  

# 定义线性回归模型
def linreg(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w) + b  # torch.Size([n, 1])

# 定义损失函数
def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # torch.Size([n, 1]), y.view(y_hat.size())将y的形状变为与y_hat相同

# 定义优化算法
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():  # 不需要计算梯度
        for param in params:
            param -= lr * param.grad / batch_size  # torch.Size([2, 1])
            param.grad.zero_()  # 梯度清零

if __name__ == "__main__":
    # 定义初始化模型参数
    w=torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # torch.Size([2, 1])
    b=torch.zeros(1, requires_grad=True)  # torch.Size([1]

    # 定义超参数
    lr = 0.03  # 学习率
    num_epochs = 3  # 迭代次数
    net=linreg  # 模型
    loss=squared_loss  # 损失函数
    bathch_size=10

    true_w=torch.tensor([2, -3.4])
    true_b=4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 训练模型
    for epoch in range(num_epochs):
        for X, y in data_iter(features, labels, bathch_size):
            # 计算梯度
            y_hat = net(X, w, b)  # torch.Size([10, 1])
            l = loss(y_hat, y)  # torch.Size([10, 1])
            l.sum().backward()  # 和backward(torch.ones(10))等价
            sgd([w, b], lr, bathch_size)  # 更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)  # torch.Size([1000, 1])
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # float()将tensor转换为float
