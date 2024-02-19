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



class EIM():
    def __init__(self,eps_treshold=0,m_basis=np.inf):
        self.m_basis=m_basis
        self.treshold=eps_treshold
        self.indices=np.array([],dtype=np.int64)

    def fit(self,y):
        m=0
        m_max=y.shape[0]
        epsilons=np.inf
        remaining_indices=np.arange(m_max)
        condition=lambda x,y: (x>self.treshold and y<m_max and y<self.m_basis)

        while condition(epsilons,m):
            error_old=np.inf
            for i in trange(m_max):
                if i in remaining_indices:
                    self.indices=np.append(self.indices,i)
                    _,residulas,_,_=np.linalg.lstsq(y[self.indices].T,y.T)
                    residulas=np.linalg.norm(residulas)
                    if residulas<error_old:
                        error_old=residulas
                        current_index=i
                    self.indices=self.indices[:-1]
            self.indices=np.append(self.indices,current_index)
            remaining_indices=remaining_indices[remaining_indices!=current_index]
            m+=1
            epsilons=error_old
            print(epsilons)        
        self.basis=y[self.indices]
    
    def transform(self,y):
        return np.linalg.lstsq(self.basis.T,y.T)[0].T
    
    def inverse_transform(self,y):
        return np.dot(y,self.basis)
    
    def get_indices(self):
        return self.indices

    def get_basis(self):
        return self.basis


class REIM():
    def __init__(self,m_basis,num_times,threshold):
        self.indices=np.array([],dtype=np.int64)
        self.threshold=threshold
        self.m_basis=m_basis
        self.num_times=num_times
    
    def fit(self,y):
        threshold_tmp=np.inf
        m_max=y.shape[0]
        m=np.min([self.m_basis,m_max])
        self.m_basis=m

        for i in trange(self.num_times):
            indexes=np.random.permutation(m_max)[:m]
            _,residulas,_,_=np.linalg.lstsq(y[indexes].T,y.T)
            if np.linalg.norm(residulas)<threshold_tmp:
                threshold_tmp=np.linalg.norm(residulas)
                self.indices=indexes
            if threshold_tmp/np.linalg.norm(y.T)<self.threshold:
                break
        
        self.basis=y[self.indices]

    def transform(self,y):
        return np.linalg.lstsq(self.basis.T,y.T)[0].T
    
    def inverse_transform(self,y):
        return np.dot(y,self.basis)
    
    def get_indices(self):
        return self.indices

    def get_basis(self):
        return self.basis







class Model():
    def __init__(self,m,threshold=0.001):
        self.model_trunk=RBF(neighbors=10)
        self.model_branch=RBF()
        self.eim=REIM(m,100,threshold)

    def fit(self,x,y,points):
        self.eim.fit(y.T)
        y_red=self.eim.transform(y.T)
        x_red=x[:,self.eim.get_indices()]
        self.model_branch.fit(x_red,self.eim.get_basis().T)
        self.model_trunk.fit(points,y_red)
    def predict(self,x,points):
        x_red=x[:,self.eim.get_indices()]
        return self.model_branch.predict(x_red)@(self.model_trunk.predict(points).T)


def norm(u):
    return u@sc_matrix@u.T



model=Model(100,0.01)
model.fit(a_train.numpy(),u_train.numpy(),points.numpy())
u_pred_test=model.predict(a_test.numpy(),points.numpy())
u_pred_train=model.predict(a_train.numpy(),points.numpy())

u_train=u_train*(b_max-b_min)+b_min
u_test=u_test*(b_max-b_min)+b_min
u_pred_test=u_pred_test*(b_max-b_min)+b_min
u_pred_train=u_pred_train*(b_max-b_min)+b_min



loss_train=np.mean(np.sqrt(np.diag(norm(u_train.numpy()-u_pred_train)/np.diag(norm(u_train.numpy())))))
loss_test=np.mean(np.sqrt(np.diag(norm(u_test.numpy()-u_pred_test)/np.diag(norm(u_test.numpy())))))

print(loss_train)
print(loss_test)


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
fig1.savefig('rbf.png')