import torch

x=torch.tensor([[1,2,3,4,5],
                [5,4,3,2,1]])
y=torch.sum(x, dim=1)
print("y",y)