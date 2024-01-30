from neuralop.models import FNO2d
from darcyflowcircle import DarcyFlowCircle
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
a=DarcyFlowCircle(10)
triangles=a.triangles
sc_matrix=a.sc_matrix
points=a.points
train_dataloader=a.train_loader
test_dataloader=a.test_loader
len_points=len(points)
_,a,b=a.train_dataset.tensors
a_max=torch.max(a)
a_min=torch.min(a)
b_max=torch.max(b)
b_min=torch.min(b)

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
        x=torch.sum(self.branch_net(x).unsqueeze(1)*self.trunk_net(points).unsqueeze(0),axis=2)
        return x
    
    def forward_eval(self, x, points):
        x=(x-a_min)/(a_max-a_min)
        x=torch.sum(self.branch_net(x).unsqueeze(1)*self.trunk_net(points).unsqueeze(0),axis=2)
        return x*(b_max-b_min)+b_min
    
index_subset=torch.randperm(len_points)[:1000]

model=DeepONet(1000,20)
Epochs=100
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.MSELoss()
sup_m=0
low_m=0
for epoch in trange(Epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        _,v,u=batch
        v=(v-a_min)/(a_max-a_min)
        u=(u-b_min)/(b_max-b_min)
        v=v.reshape(v.shape[0],-1)
        u=u.reshape(u.shape[0],-1)
        pred=model(v[:,index_subset],points)
        pred=pred.reshape(pred.shape[0],-1)
        loss=torch.linalg.norm(pred-u)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sup_m+=torch.sum(torch.linalg.norm(pred-u,axis=1)**2)/(len(train_dataloader))
            low_m+=torch.sum(torch.linalg.norm(u,axis=1)**2)/(len(train_dataloader))
    with torch.no_grad():
        print(f'Epoch: {epoch}, Loss: {torch.sqrt(sup_m/low_m)}')
        print(np.var(pred.numpy()))
        print(np.var(u.numpy()))

model=model.eval()
for batch in test_dataloader:
    _,v,u=batch
    pred=model.forward_eval(v[:,index_subset],points)
    pred=pred.reshape(pred.shape[0],-1)

def norm(u):
    return u@sc_matrix@u.T

train_rel_loss=0
sup_train_loss=0
low_train_loss=0
with torch.no_grad():
    for batch in train_dataloader:
        _,v,u=batch
        pred=model.forward_eval(v[:,index_subset],points)
        u=u.reshape(u.shape[0],-1)
        pred=pred.reshape(pred.shape[0],-1)
        u=u.numpy()
        pred=pred.numpy()
        train_rel_loss+=np.mean(np.sqrt(np.diag(norm(u-pred)/np.diag(norm(u)))))/len(train_dataloader)

print(train_rel_loss)

test_rel_loss=0
sup_test_loss=0
low_test_loss=0
with torch.no_grad():
    for batch in test_dataloader:
        _,v,u=batch
        pred=model.forward_eval(v[:,index_subset],points)
        u=u.reshape(u.shape[0],-1)
        pred=pred.reshape(pred.shape[0],-1)
        u=u.numpy()
        pred=pred.numpy()
        test_rel_loss+=np.mean(np.sqrt(np.diag(norm(u-pred)/np.diag(norm(u)))))/len(test_dataloader)


print(test_rel_loss)


import matplotlib.pyplot as plt
import matplotlib.tri as tri

triang=tri.Triangulation(points[:,0],points[:,1],triangles)
fig1, ax1 = plt.subplots(1,2)
ax1[0].set_aspect('equal')
ax1[0].set_title('Real')
tpc = ax1[0].tripcolor(triang, u[-1], shading='flat',vmin=b_min,vmax=b_max)
ax1[1].set_aspect('equal')
ax1[1].set_title('DeepONet')
tpc = ax1[1].tripcolor(triang, pred[-1], shading='flat',vmin=b_min,vmax=b_max)
#fig1.colorbar(tpc)
fig1.savefig('deeponet.png')