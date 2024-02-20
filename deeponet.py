from darcyflowcircle import DarcyFlowCircle
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
a=DarcyFlowCircle(10)
triangles=a.triangles
sc_matrix=a.sc_matrix
points=a.points.cuda()
train_dataloader=a.train_loader
test_dataloader=a.test_loader
len_points=len(points)
a,b=a.train_dataset.tensors
a_max=torch.max(a).item()
a_min=torch.min(a).item()
b_max=torch.max(b).item()
b_min=torch.min(b).item()

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

model=DeepONet(1000,20).cuda()
Epochs=100
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
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
        loss=torch.linalg.norm(pred-u)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sup_m+=torch.sum(torch.linalg.norm(pred-u,axis=1)**2)/(len(train_dataloader))
            low_m+=torch.sum(torch.linalg.norm(u,axis=1)**2)/(len(train_dataloader))
    with torch.no_grad():
        print(f'Epoch: {epoch}, Loss: {torch.sqrt(sup_m/low_m)}')
        print(torch.var(pred))
        print(torch.var(u))

model=model.eval()

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
        pred=model.forward_eval(v[:,index_subset],points)
        u=u.reshape(u.shape[0],-1)
        pred=pred.reshape(pred.shape[0],-1)
        u=u.cpu().numpy()
        pred=pred.cpu().numpy()
        u_bar=(u-b_min)/(b_max-b_min)
        pred_bar=(pred-b_min)/(b_max-b_min)
        train_rel_loss+=np.mean(np.sqrt(np.diag(norm(u_bar-pred_bar))/np.diag(norm(u_bar+1))))/len(train_dataloader)

print(train_rel_loss)

test_rel_loss=0
sup_test_loss=0
low_test_loss=0
with torch.no_grad():
    for batch in test_dataloader:
        v,u=batch
        v=v.cuda()
        u=u.cuda()
        pred=model.forward_eval(v[:,index_subset],points)
        u=u.reshape(u.shape[0],-1)
        pred=pred.reshape(pred.shape[0],-1)
        u=u.cpu().numpy()
        pred=pred.cpu().numpy()
        u_bar=(u-b_min)/(b_max-b_min)
        pred_bar=(pred-b_min)/(b_max-b_min)
        test_rel_loss+=np.mean(np.sqrt(np.diag(norm(u_bar-pred_bar))/np.diag(norm(u_bar+1))))/len(test_dataloader)

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
fig1.savefig('deeponet.png')