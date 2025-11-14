#%%
import torch
from pykeops.torch import LazyTensor
import jax
import jax.numpy as jnp

#%%

N, M, d = 3, 4, 2 
sigma = 0.5

x_torch = torch.randn(N, d)
y_torch = torch.randn(M, d)

print("x_torch =", x_torch)
print("x_torch.shape =", x_torch.shape)
print("Type :", type(x_torch))
print("----------------------------")

# initialiser d'abord avec NumPy
x_np = np.random.randn(N, d).astype(np.float32)
y_np = np.random.randn(M, d).astype(np.float32)

# puis créer les tenseurs Torch à partir de NumPy
x_torch = torch.from_numpy(x_np)
y_torch = torch.from_numpy(y_np)

# enfin créer les tableaux JAX à partir de NumPy
x_jax = jnp.array(x_np)
y_jax = jnp.array(y_np)

print("x_jax =", x_jax)
print("x_jax.shape =", x_jax.shape)
print("Type :", type(x_jax))

#%%
print("\n### PyTorch ###")
D_ij_torch = ((x_torch[:, None, :] - y_torch[None, :, :]) ** 2).sum(-1)

print("D_ij_torch =", D_ij_torch)
print("Shape :", D_ij_torch.shape)
print("Type :", type(D_ij_torch))
print("Mémoire totale (approx) :", D_ij_torch.numel() * D_ij_torch.element_size(), "octets")
print("PyTorch calcule tout de suite et stocke la matrice entière (N×M).")

#%%
print("\n### KeOps ###")

x_i = LazyTensor(x_torch[:, None, :])
y_j = LazyTensor(y_torch[None, :, :])

print("x_i =", x_i)
print("Type :", type(x_i))
print("Shape logique :", (N, 1, d))

# Exemple : création d’une expression
D_ij = ((x_i - y_j) ** 2).sum(-1)
print("\nExpression D_ij =", D_ij)
print("Type :", type(D_ij))

# Évaluation réelle
D_ij_eval = D_ij.sum(1)  # cette ligne lance le calcul GPU/C++ par blocs
print("\nÉvaluation D_ij.sum(1) =", D_ij_eval)
print("Type réel :", type(D_ij_eval))
print("Shape réelle :", D_ij_eval.shape)
print("KeOps compile et évalue le résultat sans créer de matrice N×M complète.")

#%%
print("\n### JAX ###")

D_ij_jax = jnp.sum((x_jax[:, None, :] - y_jax[None, :, :]) ** 2, axis=-1)
print("D_ij_jax =", D_ij_jax)
print("Shape :", D_ij_jax.shape)
print("Type :", type(D_ij_jax))
print("JAX évalue comme NumPy, mais peut être compilé avec JIT.")

# Compilation avec XLA
@jax.jit # JIT : Just-In-Time compilation
def gaussianconv_jax(X, Y, S):
    D_ij = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    K_ij = jnp.exp(-D_ij / (2 * S**2))
    return jnp.sum(K_ij, axis=1)

print("\nCompilation JAX avec JIT...")
f_jax = gaussianconv_jax(x_jax, y_jax, sigma)
print("f_jax =", f_jax)
print("Type après JIT :", type(f_jax))
print("Shape :", f_jax.shape)
print("JAX trace le graphe, le compile en code bas-niveau (XLA) puis exécute.")

#%%
print("""
Bilan final - comparaison Pytorch vs KeOps vs JAX :      
      
PyTorch :
    - Type : torch.Tensor
    - Évaluation : immédiate
    - Stocke toute la matrice N×M
    - Bon compromis entre flexibilité et performance

KeOps :
    - Type : LazyTensor
    - Évaluation : paresseuse (lazy)
    - Stocke uniquement les données nécessaires (x, y)
    - Crée un kernel C++/CUDA sur mesure et travaille par blocs
    - Énorme gain mémoire sur grands N et M

JAX :
    - Type : jax.Array (DeviceArray)
    - Évaluation : compilée (via JIT)
    - Trace le graphe et fusionne les opérations
    - Idéal pour différentiation et exécution sur GPU/TPU
    - Mais stocke tout le calcul en mémoire pendant exécution
""")

# %%

import torch
import jax
import jax.numpy as jnp
from pykeops.torch import LazyTensor
import time
import psutil
import os
import numpy as np


def ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6  # en Mo



print("\n### Données communes ###")
N, M, d = 5000, 5000, 2  
sigma = 0.3

x_torch = torch.randn(N, d)
y_torch = torch.randn(M, d)
x_jax = jnp.array(x_torch.numpy())
y_jax = jnp.array(y_torch.numpy())

print(f"x_torch.shape = {x_torch.shape}, type = {type(x_torch)}")
print(f"x_jax.shape   = {x_jax.shape}, type = {type(x_jax)}")


print("\n### PyTorch ###")

start_ram = ram_usage()
start = time.time()

D_ij_torch = ((x_torch[:, None, :] - y_torch[None, :, :]) ** 2).sum(-1)
K_ij_torch = (- D_ij_torch / (2 * sigma**2)).exp()
f_torch = K_ij_torch.sum(dim=1)

end = time.time()
end_ram = ram_usage()

torch_time = end - start
torch_ram = end_ram - start_ram

print(f"Temps PyTorch : {torch_time:.3f} s")
print(f"RAM utilisée : {torch_ram:.2f} Mo")
print(f"Résultat f_torch.shape = {f_torch.shape}")

print("\n### JAX (compilation JIT) ###")

@jax.jit
def gaussianconv_jax(X, Y, S):
    D_ij = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    K_ij = jnp.exp(-D_ij / (2 * S**2))
    return jnp.sum(K_ij, axis=1)


start_ram = ram_usage()
start = time.time()
f_jax = gaussianconv_jax(x_jax, y_jax, sigma).block_until_ready() # block_until_ready pour s'assurer que le calcul est terminé
end = time.time()
end_ram = ram_usage()

jax_compile_time = end - start
jax_compile_ram = end_ram - start_ram

start = time.time()
f_jax2 = gaussianconv_jax(x_jax, y_jax, sigma).block_until_ready()
end = time.time()

jax_run_time = end - start

print(f"Temps JAX (1er run = compilation) : {jax_compile_time:.3f} s")
print(f"Temps JAX (2e run) : {jax_run_time:.3f} s")
print(f"RAM utilisée (1er run) : {jax_compile_ram:.2f} Mo")
print(f"Résultat f_jax.shape = {f_jax.shape}")


print("\n### KeOps (LazyTensor) ###")

x_i = LazyTensor(x_torch[:, None, :])
y_j = LazyTensor(y_torch[None, :, :])
D_ij = ((x_i - y_j) ** 2).sum(-1)
K_ij = (- D_ij / (2 * sigma**2)).exp()

start_ram = ram_usage()
start = time.time()
f_keops = K_ij.sum(dim=1)  # Lancement du calcul
end = time.time()
end_ram = ram_usage()

keops_time = end - start
keops_ram = end_ram - start_ram

print(f"Temps KeOps : {keops_time:.3f} s")
print(f"RAM utilisée : {keops_ram:.2f} Mo")
print(f"Résultat f_keops.shape = {f_keops.shape}")



# %%