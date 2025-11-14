import torch
import matplotlib.pyplot as plt
from pykeops.torch import LazyTensor
import jax
import jax.numpy as jnp

#%%
device = torch.device("cpu")
dtype = torch.float32

N, M, d = 3000, 3000, 2
sigma = 0.3

#%%
x_torch = torch.randn(N, d, device=device, dtype=dtype)
y_torch = torch.randn(M, d, device=device, dtype=dtype)

x_jax = jnp.array(x_torch.numpy())
y_jax = jnp.array(y_torch.numpy())

#%%
def gaussianconv_torch(X, Y, S):
    D_ij = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    K_ij = (- D_ij / (2 * S**2)).exp()
    f = K_ij.sum(dim=1)
    return f.view(X.shape[0], 1)

#%%
def gaussianconv_keops(X, Y, S):
    x_i = LazyTensor(X[:, None, :])
    y_j = LazyTensor(Y[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    K_ij = (- D_ij / (2 * S**2)).exp()
    f = K_ij.sum(dim=1)
    return f.view(X.shape[0], 1)

#%%
@jax.jit # JIT compilation : Just-In-Time compilation avec XLA
def gaussianconv_jax(X, Y, S):
    D_ij = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    K_ij = jnp.exp(-D_ij / (2 * S**2))
    f = jnp.sum(K_ij, axis=1)
    return f.reshape((X.shape[0], 1))

#%%
f_torch = gaussianconv_torch(x_torch, y_torch, sigma)
f_keops = gaussianconv_keops(x_torch, y_torch, sigma)
f_jax   = gaussianconv_jax(x_jax, y_jax, sigma)

#%%
f_jax_torch = torch.tensor(jnp.array(f_jax), dtype=torch.float32)

print("Différence Torch vs KeOps :", torch.max(torch.abs(f_torch - f_keops)).item())
print("Différence Torch vs JAX   :", torch.max(torch.abs(f_torch - f_jax_torch)).item())

assert torch.allclose(f_torch, f_keops, atol=1e-5)
assert torch.allclose(f_torch, f_jax_torch, atol=1e-5)