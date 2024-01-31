import torch
from torch import nn
from torch.func import vmap,hessian
points=torch.rand(400,2)
v=torch.rand(10,300)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin=nn.Linear(300,1)
    def forward(self,x,points):
        return self.lin(x)*torch.linalg.norm(points)
    

model=NN()


def fun(x,points):
    return model(x,points).squeeze(0)

print(fun(v,points).shape)

print(v[0].shape)
print(points.shape)
print(hessian(fun,argnums=1)(v,points).shape)

def laplacian(f,x,points):
  return torch.diagonal(vmap(vmap(hessian(f,argnums=1),in_dims=(None,0)),in_dims=(0,None))(x,points),dim1 = -2, dim2 = -1).sum(axis=2)
print(laplacian(fun,v,points).shape)