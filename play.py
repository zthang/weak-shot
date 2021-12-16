import torch
from torch_scatter import scatter

a=torch.tensor([[1,1,1],
                [2,2,2],
                [4,4,4]])
b=scatter(a, torch.tensor([0, 0, 2]), dim=0)
print(1)