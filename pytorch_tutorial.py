import torch
import math

dtype=torch.float
device = torch.device("cpu")
x=torch.tensor([3,4,6,7])
y=torch.tensor([1,2,3,4])
z=x+y
print(z)

