import numpy as np
from ezyrb import RBF, GPR, POD
from sklearn.tree import DecisionTreeRegressor
import scipy.linalg


from darcyflowcircle import DarcyFlowCircle
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
a=DarcyFlowCircle(10)
triangles=a.triangles
sc_matrix=a.sc_matrix
points=a.points
a_train,u_train=a.train_dataset.tensors
a_test,u_test=a.test_dataset.tensors


a_min=torch.min(a_train).item()
a_max=torch.max(a_train).item()
b_min=torch.min(u_train).item()
b_max=torch.max(u_test).item()

a_train=(a_train-a_min)/(a_max-a_min)
u_train=(u_train-b_min)/(b_max-b_min)
a_test=(a_test-a_min)/(a_max-a_min)
u_test=(u_test-b_min)/(b_max-b_min)


class Operator():
    def __init__(self,m,p): #m is the number of sensors
        self.model_trunk=DecisionTreeRegressor(max_depth=100)
        self.model_branch=RBF()
        self.pod=POD(method='randomized_svd',rank=p)
        self.m=m
        self.p=p
    def fit(self,x,y,points):
        self.pod.fit(y)
        self.modes=self.pod.modes
        self.basis=self.pod.transform(y)
        self.indices=np.random.permutation(self.basis.shape[1])[:self.m]
        self.model_branch.fit(x[:,self.indices],self.modes)
        self.model_trunk.fit(points,self.basis.T)

    def predict(self,x,points):
        x_red=x[:,self.indices]
        return self.model_branch.predict(x_red)@(self.model_trunk.predict(points).T)
    


def norm(u):
    return u@sc_matrix@u.T



model=Operator(p=100,m=150)
model.fit(a_train.numpy(),u_train.numpy(),points.numpy())

u_pred_test=model.predict(a_test.numpy(),points.numpy())
u_pred_train=model.predict(a_train.numpy(),points.numpy())

u_train=u_train*(b_max-b_min)+b_min
u_test=u_test*(b_max-b_min)+b_min
u_pred_test=u_pred_test*(b_max-b_min)+b_min
u_pred_train=u_pred_train*(b_max-b_min)+b_min

u_bar_train=(u_train-b_min)/(b_max-b_min)
pred_bar_train=(u_pred_train-b_min)/(b_max-b_min)
u_bar_test=(u_test-b_min)/(b_max-b_min)
pred_bar_test=(u_pred_test-b_min)/(b_max-b_min)

loss_train=np.mean(np.sqrt(np.diag(norm(u_bar_train.numpy()-pred_bar_train))/np.diag(norm(u_bar_train.numpy()+1))))
loss_test=np.mean(np.sqrt(np.diag(norm(u_bar_test.numpy()-pred_bar_test))/np.diag(norm(u_bar_test.numpy()+1))))


import matplotlib.pyplot as plt
import matplotlib.tri as tri

triang=tri.Triangulation(points[:,0],points[:,1],triangles)
fig1, ax1 = plt.subplots(1,2)
ax1[0].set_aspect('equal')
ax1[0].set_title('Real')
tpc = ax1[0].tripcolor(triang, u_test[-1], shading='flat',vmin=b_min,vmax=b_max)
ax1[1].set_aspect('equal')
ax1[1].set_title('EIM+RBF')
tpc = ax1[1].tripcolor(triang, u_pred_test[-1], shading='flat',vmin=b_min,vmax=b_max)
#fig1.colorbar(tpc)
fig1.savefig('tree_pod.png')

print(loss_train)
print(loss_test)