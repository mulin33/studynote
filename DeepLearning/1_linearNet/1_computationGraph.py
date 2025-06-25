import torch

# 标量对标量的求导是一个值
x=torch.tensor(3.0, requires_grad=True)  
print(x.data)
print(x.grad)  # None
print(x.is_leaf)  # True
print(x.requires_grad)  # True

y=6*x**2+2*x+4
print(y.data)
print(y.retain_grad())  # None .grad属性只能在叶子节点上使用
print(y.is_leaf)  # False
print(y.requires_grad)  # True

y.backward()  # 反向传播
print(x.grad)  # 关于x的导数：tensor(38.)
print(y.grad)   # 关于y的导数：tensor(1.) ，一般仍为None

# 标量对向量的求导是一个向量
x_vec=torch.arange(4.0, requires_grad=True)  
y_vec=2*torch.dot(x_vec,x_vec)  # 2*(0^2+1^2+2^2+3^2)
print(y_vec.data)   # tensor(28.)

y_vec.backward()  
print(x_vec.grad)  # tensor([0., 4., 8., 12.])

# pytorh需要手动清空梯度，否则会将新的梯度累加（加上）到已有的梯度上，而不是覆盖它
x_vec.grad.zero_()  # 清空梯度,如果不清空，PyTorch 
y_vec=x_vec.sum()  # y=0+1+2+3
print(y_vec.data)  # tensor(6.)

y_vec.backward()
print(x_vec.grad)  # tensor([1., 1., 1., 1.])，如果不清空梯度为tensor([1.,5.,9.,13.])

# 向量对向量的求导是一个矩阵
x_vec.grad.zero_()  
y_vec=x_vec*x_vec
print(y_vec.data)  # tensor([0., 1., 4., 9.])

y_vec.backward(torch.ones(len(x_vec)))  # 传入一个与y_vec形状相同的向量(反向传播起点的梯度值（通常是全 1），类似加权梯度的作用)
print(x_vec.grad)  # tensor([0., 2., 4., 6.])