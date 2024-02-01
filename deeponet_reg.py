from darcyflowcircle import DarcyFlowCircle
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
from torch.func import vmap,hessian
from time import time
a=DarcyFlowCircle(5)
triangles=a.triangles
sc_matrix=a.sc_matrix
points=a.points.cuda()
train_dataloader=a.train_loader
test_dataloader=a.test_loader
len_points=len(points)
a,b=a.train_dataset.tensors
a_max=torch.max(a).cuda()
a_min=torch.min(a).cuda()
b_max=torch.max(b).cuda()
b_min=torch.min(b).cuda()

class DeepONet(nn.Module):
    def __init__(self,
                num_start_points,
                medium_size
                ):
    
        super().__init__()
        self.num_start_points=num_start_points
        self.medium_size=medium_size
        self.branch_net=nn.Sequential(nn.Linear(num_start_points,100), nn.ReLU(),nn.Linear(100,100), nn.ReLU(),nn.Linear(100,100), nn.ReLU(), nn.Linear(100,medium_size))
        self.trunk_net=nn.Sequential(nn.Linear(2,100), nn.ReLU(),nn.Linear(100,100), nn.ReLU(),nn.Linear(100,100), nn.ReLU(), nn.Linear(100,medium_size))
    
    def forward(self, x, points):
        x=(0.45-torch.linalg.norm(points-0.5,axis=-1))*torch.sum(self.branch_net(x).unsqueeze(1)*self.trunk_net(points).unsqueeze(0),axis=2)
        return x
    
    def forward_eval(self, x, points):
        x=(x-a_min)/(a_max-a_min)
        x=(0.45-torch.linalg.norm(points-0.5,axis=-1))*torch.sum(self.branch_net(x).unsqueeze(1)*self.trunk_net(points).unsqueeze(0),axis=2)
        return x*(b_max-b_min)+b_min
    


class SimpleDeepONet(nn.Module):
    def __init__(self,
                num_start_points,
                medium_size
                ):
    
        super().__init__()
        self.num_start_points=num_start_points
        self.medium_size=medium_size
        self.branch_net=nn.Sequential(nn.Linear(num_start_points,100), nn.ReLU(),nn.Linear(100,100), nn.ReLU(),nn.Linear(100,100), nn.ReLU(), nn.Linear(100,medium_size))
        self.trunk_net=nn.Sequential(nn.Linear(2,100), nn.ReLU(),nn.Linear(100,100), nn.ReLU(),nn.Linear(100,100), nn.ReLU(), nn.Linear(100,medium_size))
    
    def forward(self, x, points):
        x=(0.45-torch.linalg.norm(points-0.5))*torch.sum(self.branch_net(x)*self.trunk_net(points))
        return x
    
    def forward_eval(self, x, points):
        x=(x-a_min)/(a_max-a_min)
        x=(0.45-torch.linalg.norm(points-0.5))*torch.sum(self.branch_net(x)*self.trunk_net(points))
        return x*(b_max-b_min)+b_min



def laplacian(f,x,points):
  return torch.diagonal(vmap(vmap(hessian(f,argnums=1),in_dims=(None,0)),in_dims=(0,None))(x,points),dim1 = -2, dim2 = -1).sum(axis=2)

index_subset=torch.randperm(len_points)[:1000].cuda()


non_vec_model=SimpleDeepONet(1000,20).cuda()

def model_1(x,points):
    return torch.vmap(non_vec_model,in_dims=(None,0))(x,points)

def model(x,points):
    return torch.vmap(model_1,in_dims=(0,None))(x,points)

def model_1_eval(x,points):
    return torch.vmap(non_vec_model.forward_eval,in_dims=(None,0))(x,points)

def model_eval(x,points):
    return torch.vmap(model_1_eval,in_dims=(0,None))(x,points)


Epochs=10
optimizer=torch.optim.Adam(non_vec_model.parameters(),lr=0.001)
loss_fn=nn.MSELoss()
sup_m=0
low_m=0
for epoch in trange(Epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        v,u=batch
        v=v.cuda()
        u=u.cuda()
        v=(v-a_min)/(a_max-a_min)
        u=(u-b_min)/(b_max-b_min)
        v=v.reshape(v.shape[0],-1)
        u=u.reshape(u.shape[0],-1)
        pred=model(v[:,index_subset],points)
        pred=pred.reshape(pred.shape[0],-1)
        loss=torch.linalg.norm(pred-u)+0.01*torch.linalg.norm(v*laplacian(non_vec_model,v[:,index_subset],points)-1)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sup_m+=torch.sum(torch.linalg.norm(pred-u,axis=1)**2)/(len(train_dataloader))
            low_m+=torch.sum(torch.linalg.norm(u,axis=1)**2)/(len(train_dataloader))
    with torch.no_grad():
        print(f'Epoch: {epoch}, Loss: {torch.sqrt(sup_m/low_m)}')
        print(torch.var(pred))
        print(torch.var(u))

def norm(u):
    return u@sc_matrix@u.T

train_rel_loss=0
sup_train_loss=0
low_train_loss=0
with torch.no_grad():
    for batch in train_dataloader:
        v,u=batch
        v=v.cuda()
        u=u.cuda()
        pred=model_eval(v[:,index_subset],points)
        u=u.reshape(u.shape[0],-1)
        pred=pred.reshape(pred.shape[0],-1)
        u=u.cpu().numpy()
        pred=pred.cpu().numpy()
        train_rel_loss+=np.mean(np.sqrt(np.diag(norm(u-pred)/np.diag(norm(u)))))/len(train_dataloader)

print(train_rel_loss)

test_rel_loss=0
sup_test_loss=0
low_test_loss=0
with torch.no_grad():
    for batch in test_dataloader:
        v,u=batch
        v=v.cuda()
        u=u.cuda()
        pred=model_eval(v[:,index_subset],points)
        u=u.reshape(u.shape[0],-1)
        pred=pred.reshape(pred.shape[0],-1)
        u=u.cpu().numpy()
        pred=pred.cpu().numpy()
        test_rel_loss+=np.mean(np.sqrt(np.diag(norm(u-pred)/np.diag(norm(u)))))/len(test_dataloader)


print(test_rel_loss)


import matplotlib.pyplot as plt
import matplotlib.tri as tri
points=points.cpu().numpy()
triang=tri.Triangulation(points[:,0],points[:,1],triangles)
fig1, ax1 = plt.subplots(1,2)
ax1[0].set_aspect('equal')
ax1[0].set_title('Real')
tpc = ax1[0].tripcolor(triang, u[-1], shading='flat',vmin=b_min,vmax=b_max)
ax1[1].set_aspect('equal')
ax1[1].set_title('DeepONet')
tpc = ax1[1].tripcolor(triang, pred[-1], shading='flat',vmin=b_min,vmax=b_max)
#fig1.colorbar(tpc)
fig1.savefig('deeponet_reg.png')