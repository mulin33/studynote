import torch
from torch import nn
import sys
import os
sys.path.append(os.path.abspath("1_linearNet"))  
import softmaxClassify


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)  # Reshape to (batch_size, channels, height, width)
    
net = nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), 
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))


batch_size = 256
train_iter, test_iter = softmaxClassify.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy(data_iter, net, device=None):
    """Evaluate the accuracy of a model on gpu."""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device    #  如果没有指定 device，就自动从模型参数中获取
    metric = softmaxClassify.Accumulator(2)  # Correct predictions, total predictions
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]  # Handle multiple inputs
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(softmaxClassify.accuracy(net(X), y), y.numel())
    print("evaluate on", device)
    return metric[0] / metric[1]  # Return accuracy

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)  # Initialize weights
    print("Training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = softmaxClassify.Animator(xlabel='epoch', ylabel='loss',
                                        legend=['train loss', 'train acc', 'test acc'])
    
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        metric = softmaxClassify.Accumulator(3)  # Train loss, train acc, test acc
        net.train()
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], softmaxClassify.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0:
                animator.add(epoch + (i + 1) / num_batches, (train_loss, train_acc, None))
        test_acc = evaluate_accuracy(test_iter, net, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')

if __name__ == '__main__':
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs = 0.9, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    softmaxClassify.stick_show()  