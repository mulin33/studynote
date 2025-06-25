import torch 

x=torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

x=x.reshape(3,4)
print(x)
print(x.shape)

true_w = torch.tensor([2, -3.4])    # torch.Size([2])
print(true_w.shape)
print(len(true_w))
row_vec=torch.tensor([[2,-3.4]])    #torch.Size([1, 2])
print(row_vec.shape)
print(len(row_vec))
col_vec=torch.tensor([[2],[-3.4]])  #torch.Size([2, 1])
print(col_vec.shape)
print(len(col_vec))