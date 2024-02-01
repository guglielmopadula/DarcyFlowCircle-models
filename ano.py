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
a_max=torch.max(a).cuda()
a_min=torch.min(a).cuda()
b_max=torch.max(b).cuda()
b_min=torch.min(b).cuda()

#Averaging Neural Operator
class ANOLayer(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.lin=nn.Linear(hidden_size,hidden_size)

    def forward(self,x):
        return self.lin(x)+torch.mean(x,axis=1).unsqueeze(1)
    
class LAR(nn.Module):  
    def __init__(self,hidden_size):
        super().__init__()
        self.ano=ANOLayer(hidden_size)
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(hidden_size)
        
    def forward(self,x):
        x=self.ano(x)
        s=x.shape[1]
        n=x.shape[0]
        x=x.reshape(x.shape[0]*s,-1)
        x=self.bn(x)
        x=x.reshape(n,s,-1)
        return self.relu(x)

class ANO(nn.Module):
    def __init__(self,input_size,output_size,hidden_size,points,BATCH_SIZE):
    
        super().__init__()
        self.R=nn.Linear(input_size+2,hidden_size)
        self.hidden_layers=nn.Sequential(LAR(hidden_size),LAR(hidden_size),LAR(hidden_size))
        self.Q=nn.Linear(hidden_size+2,output_size)
        self.points=points.unsqueeze(0).repeat(BATCH_SIZE,1,1)
    def forward(self, x):
        x=self.R(torch.cat((x,self.points),2))
        x=self.hidden_layers(x)
        x=self.Q(torch.cat((x,self.points),2))
        return x
    
    def forward_eval(self, x):
        x=(x-a_min)/(a_max-a_min)
        x=self.R(torch.cat((x,self.points),2))
        x=self.hidden_layers(x)
        x=self.Q(torch.cat((x,self.points),2))
        return x*(b_max-b_min)+b_min
    


    
model=ANO(1,1,100,points,10).cuda()
Epochs=10
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)    
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
        v=v.reshape(v.shape[0],-1,1)
        u=u.reshape(u.shape[0],-1)
        pred=model(v)
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
        v=v.reshape(v.shape[0],-1,1)
        pred=model.forward_eval(v)
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
        v=v.reshape(v.shape[0],-1,1)
        pred=model.forward_eval(v)
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
ax1[1].set_title('ANO')
tpc = ax1[1].tripcolor(triang, pred[-1], shading='flat',vmin=b_min,vmax=b_max)
#fig1.colorbar(tpc)
fig1.savefig('ano.png')