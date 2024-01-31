import torch
import torch.nn as nn
import torch.nn.functional as F
from darcyflowcircle import DarcyFlowCircle
import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
from time import time

# class for fully nonequispaced 2d points, using the F-FNO approach
class FVFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions) + 1)
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions) + 1)
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.arange(modes).repeat(self.batch_size, 1)[:,:,None].float()
        self.Y_ = torch.arange(modes).repeat(self.batch_size, 1)[:,:,None].float()


        self.V_fwd_X, self.V_inv_X, self.V_fwd_Y, self.V_inv_Y = self.make_matrix()

    def make_matrix(self):
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:])
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]))
        forward_mat_X = torch.exp(-1j* (X_mat))
        forward_mat_Y = torch.exp(-1j* (Y_mat))

        inverse_mat_X = torch.conj(forward_mat_X).permute(0,2,1)
        inverse_mat_Y = torch.conj(forward_mat_Y).permute(0,2,1)

        return forward_mat_X, inverse_mat_X, forward_mat_Y, inverse_mat_Y

    def forward(self, data):
        fwd_X = torch.bmm(self.V_fwd_X, data)
        fwd_Y = torch.bmm(self.V_fwd_Y, data)
        return fwd_X, fwd_Y

    def inverse(self, data_x, data_y):
        inv_X = torch.bmm(self.V_inv_X, data_x)
        inv_Y = torch.bmm(self.V_inv_Y, data_y)
        return inv_X, inv_Y


class SpectralConv2d_SMM (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_SMM, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        # pdb.set_trace()

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))


    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, transformer):
        x = x.permute(0, 2, 1)
        
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft, y_ft = transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft = x_ft.permute(0, 2, 1)
        y_ft = y_ft.permute(0, 2, 1)

        # Multiply relevant Fourier modes
        out_ft_x = self.compl_mul1d(x_ft, self.weights1)
        out_ft_y = self.compl_mul1d(y_ft, self.weights1)

        #Return to physical space
        out_ft_x = out_ft_x.permute(0, 2, 1)
        out_ft_y = out_ft_y.permute(0, 2, 1)

        x, y = transformer.inverse(out_ft_x, out_ft_y) # x [4, 20, 512, 512]
        x = x+y
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real
    

class PreVNO(nn.Module):
    # Set a class attribute for the default configs.
    def __init__ (self, modes1,modes2,width,points):
        super(PreVNO, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic

        # Predictions are normalized, we need the output denormalized


        self.fc0 = nn.Linear(1, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_SMM(self.width, self.width, self.modes1, self.modes2)
        self.w01 = nn.Conv1d(self.width, self.width, 1)
        self.w02 = nn.Conv1d(self.width, self.width, 1)
        self.w11 = nn.Conv1d(self.width, self.width, 1)
        self.w12 = nn.Conv1d(self.width, self.width, 1)
        self.w21 = nn.Conv1d(self.width, self.width, 1)
        self.w22 = nn.Conv1d(self.width, self.width, 1)
        self.w31 = nn.Conv1d(self.width, self.width, 1)
        self.w32 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.transform = FVFT(points[:,:,0], points[:,:,1], self.modes1)

    def forward (self, x):
        # Elasticity has these two as inputs

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x, self.transform)
        x2 = self.w01(x1)
        x3 = F.gelu(x2)
        x4 = self.w02(x3)
        x = F.gelu(x4) + x

        x1 = self.conv1(x, self.transform)
        x2 = self.w11(x1)
        x3 = F.gelu(x2)
        x4 = self.w12(x3)
        x = F.gelu(x4) + x

        x1 = self.conv2(x, self.transform)
        x2 = self.w21(x1)
        x3 = F.gelu(x2)
        x4 = self.w22(x3)
        x = F.gelu(x4) + x

        x1 = self.conv3(x, self.transform)
        x2 = self.w31(x1)
        x3 = F.gelu(x2)
        x4 = self.w32(x3)
        x = F.gelu(x4) + x

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    


BATCH_SIZE=1
a=DarcyFlowCircle(BATCH_SIZE)
triangles=a.triangles
sc_matrix=a.sc_matrix
points=a.points.unsqueeze(0).repeat(BATCH_SIZE,1,1)
train_dataloader=a.train_loader
test_dataloader=a.test_loader
model=PreVNO(12,12,100,points)


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
_,a,b=a.train_dataset.tensors
a_max=torch.max(a)
a_min=torch.min(a)
b_max=torch.max(b)
b_min=torch.min(b)


Epochs=1
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.MSELoss()
sup_m=0
low_m=0
t=0
for epoch in trange(Epochs):
    start=time()
    for batch in train_dataloader:
        optimizer.zero_grad()
        _,v,u=batch
        v=v.reshape(v.shape[0],-1,1)
        v=(v-a_min)/(a_max-a_min)
        u=(u-b_min)/(b_max-b_min)
        u=u.reshape(u.shape[0],-1,1)
        pred=model(v)
        pred=pred.reshape(pred.shape[0],-1,1)
        loss=torch.linalg.norm(pred-u)
        loss.backward()
        optimizer.step()
        t=t+BATCH_SIZE
        if t==1:
            end=time()
            print(end-start)
        with torch.no_grad():
            sup_m+=torch.sum(torch.linalg.norm(pred-u,axis=1)**2)/(len(train_dataloader))
            low_m+=torch.sum(torch.linalg.norm(u,axis=1)**2)/(len(train_dataloader))
    with torch.no_grad():
        print(f'Epoch: {epoch}, Loss: {torch.sqrt(sup_m/low_m)}')
        print(np.var(pred.numpy()))
        print(np.var(u.numpy()))



model=model.eval()
pred=pred.reshape(pred.shape[0],-1)

def norm(u):
    return u@sc_matrix@u.T

train_rel_loss=0
sup_train_loss=0
low_train_loss=0
with torch.no_grad():
    for batch in train_dataloader:
        _,v,u=batch
        v=v.reshape(v.shape[0],-1,1)
        v=(v-a_min)/(a_max-a_min)
        pred=model.forward(v)*(b_max-b_min)+b_min
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
        v=v.reshape(v.shape[0],-1,1)
        pred=model.forward(v)*(b_max-b_min)+b_min
        u=u.reshape(u.shape[0],-1)
        pred=pred.reshape(pred.shape[0],-1)
        u=u.numpy()
        pred=pred.numpy()
        test_rel_loss+=np.mean(np.sqrt(np.diag(norm(u-pred)/np.diag(norm(u)))))/len(test_dataloader)


print(test_rel_loss)


import matplotlib.pyplot as plt
import matplotlib.tri as tri
points=points.numpy().reshape(-1,2)
triang=tri.Triangulation(points[:,0],points[:,1],triangles)
fig1, ax1 = plt.subplots(1,2)
ax1[0].set_aspect('equal')
ax1[0].set_title('Real')
tpc = ax1[0].tripcolor(triang, u[-1], shading='flat',vmin=b_min,vmax=b_max)
ax1[1].set_aspect('equal')
ax1[1].set_title('DeepONet')
tpc = ax1[1].tripcolor(triang, pred[-1], shading='flat',vmin=b_min,vmax=b_max)
#fig1.colorbar(tpc)
fig1.savefig('vno.png')