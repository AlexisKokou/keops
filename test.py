#%%
import torch
import matplotlib.pyplot as plt
from pykeops.torch import LazyTensor
import jax
import jax.numpy as jnp

#%%
device = torch.device("cpu")
dtype = torch.float32

#%%

N, M, d = 3000, 3000, 2 
x = torch.randn(N, d, device=device, dtype=dtype)
y = torch.randn(M, d, device=device, dtype=dtype)
sigma = 0.3

#%%
x_i = LazyTensor(x[:, None, :])   # (N, 1, d) -> constructeur
y_j = LazyTensor(y[None, :, :])   # (1, M, d)

#%%

def gaussianconv(X,Y,S) : 
    
    D_ij = ((X - Y) ** 2).sum(-1) #-> le sum(-1) effectue la 
#somme sur le dernier indice càd 
    K_ij = (- D_ij / (2 * S**2)).exp()
    f = K_ij.sum(dim=1) # on fait la somme sur la deuxième dimension sur les j 
    return f.view(X.shape[0], 1)
#%% 
f_torch=gaussianconv(x[:, None, :] ,y[None, :, :] ,sigma)  
f_tensor=gaussianconv(x_i ,y_j ,sigma) 

assert torch.allclose(f_torch, f_tensor)