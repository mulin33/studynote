import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, use_data=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.use_data = use_data  # 控制是否使用 .data

    def forward(self, X):
        if self.use_data:
            linear = torch.matmul(X, self.weight.data) + self.bias.data
        else:
            linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)

# 造一个简单的输入和损失
X = torch.randn(2, 5, requires_grad=True)
y = torch.randn(2, 3)

# 用不使用 .data 的两个模型对比
for use_data in [False, True]:
    model = MyLinear(5, 3, use_data=use_data)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print(f"\n=== use_data = {use_data} ===")

    out = model(X)
    loss = F.mse_loss(out, y)
    loss.backward()

    print("weight.grad:")
    print(model.weight.grad)

    print("bias.grad:")
    print(model.bias.grad)

    # 清除梯度，防止影响下一次
    optimizer.zero_grad()
