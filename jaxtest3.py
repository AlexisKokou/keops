
#%%
import numpy as np
import jax
import jax.numpy as jnp
#%%



#%%
print("jax_enable_x64 =", jax.config.read("jax_enable_x64"), "\n")
#Est-ce que tu m’autorises à utiliser le type float64 (double précision) ?
#NON !


x_default = jnp.array([1.0, 2.0, 3.0])
print("x_default =", x_default)
print("dtype =", x_default.dtype, "JAX utilise float32 par défaut\n")



x_try64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
print("dtype obtenu =", x_try64.dtype, "Tronqué vers float32\n")
#le float64 est désactivé → je convertis automatiquement en float32.



jax.config.update("jax_enable_x64", True)
#activation du float64


# Recalcul en float64
x64_after = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
print("x64_after demandé en float64 =", x64_after)
print("dtype obtenu =", x64_after.dtype, "  <-- Cette fois : float64 ✓\n")


# Par défaut, JAX désactive le support du float64
# les données sont automatiquement converties en float32,même lorsqu’on demande explicitement du float64.
#  Ensuite, on active l’option jax_enable_x64, qui autorise l’utilisation du float64. 
# Après cette activation, JAX respecte correctement les types demandés.

#%%
import numpy as np
import torch
from pykeops.torch import LazyTensor

import jax
import jax.numpy as jnp



print("Avant activation : jax_enable_x64 =", jax.config.read("jax_enable_x64"))
jax.config.update("jax_enable_x64", True)
print("Après activation : jax_enable_x64 =", jax.config.read("jax_enable_x64"), "\n")

# Convo gaussienne jax -> keops -> jax 

def gaussianconv_jax_keops(X_jax, Y_jax, sigma):

    #### 1) JAX → NumPy (float64 préservé)
    X_np = np.array(X_jax)
    Y_np = np.array(Y_jax)
    print("dtype NumPy reçu :", X_np.dtype)

    #### 2) NumPy → Torch (float64)
    X_torch = torch.from_numpy(X_np).double()   # double = float64
    Y_torch = torch.from_numpy(Y_np).double()
    print("dtype Torch utilisé :", X_torch.dtype)

    #### 3) LazyTensor KeOps
    x_i = LazyTensor(X_torch[:, None, :])     # (N,1,d)
    y_j = LazyTensor(Y_torch[None, :, :])     # (1,M,d)

    #### 4) Convolution gaussienne
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    K_ij = (- D_ij / (2 * sigma**2)).exp()
    f_torch = K_ij.sum(dim=1)                # (N,1)

    #### 5) Torch → NumPy → JAX (float64)
    f_np = f_torch.cpu().numpy()
    f_jax = jnp.array(f_np)                  # float64 car x64 activé

    return f_jax




N, M, d = 3000, 3000, 2
sigma = 0.3

x_np = np.random.randn(N, d)
y_np = np.random.randn(M, d)

x_jax = jnp.array(x_np, dtype=jnp.float64)
y_jax = jnp.array(y_np, dtype=jnp.float64)

print("x_jax dtype :", x_jax.dtype)
print("y_jax dtype :", y_jax.dtype)


x_torch = torch.tensor(x_np, dtype=torch.float64)
y_torch = torch.tensor(y_np, dtype=torch.float64)


# Convo gaussienne torch

def gaussianconv_torch(X, Y, S):
    D_ij = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    K_ij = torch.exp(-D_ij / (2 * S**2))
    f = K_ij.sum(dim=1)
    return f.view(X.shape[0], 1)


# Comparatif des deux 


f_torch = gaussianconv_torch(x_torch, y_torch, sigma)
print("f_torch.shape =", f_torch.shape)

f_jax_keops = gaussianconv_jax_keops(x_jax, y_jax, sigma)

print("f_jax_keops.shape =", f_jax_keops.shape)
print("f_jax_keops dtype =", f_jax_keops.dtype)



f_torch_np = f_torch.numpy()
f_jax_np = np.array(f_jax_keops)

max_diff = np.max(np.abs(f_torch_np - f_jax_np))

print("\nDifférence max Torch vs JAX_KeOps :", max_diff)



# %%

# Calcul de différences relatives 
eps = 1e-12

relative_diff = np.max(
    np.abs(f_torch_np - f_jax_np) / (np.abs(f_torch_np) + eps)
)

print("Différence relative Torch vs JAX_KeOps :", relative_diff)

# %%

# On va passer maintenant à la documentation de l'autodiff avec keops
# Lorsque keops va fair la phase de réduction (sum= 1) 
# keops va générer noyau automatiquement pour le backward 

# %%


x_torch_grad = x_torch.clone().requires_grad_(True)
#PyTorch doit suivre les opérations faites sur ce tensor pour calculer les gradients

f_torch_grad = gaussianconv_torch(x_torch_grad, y_torch, sigma) #calcule la convolution gaussienne
#(N,1)
f_torch_grad.sum().backward()
#PyTorch a besoin que la fonction soit scalaire pour appeler backward().

print("Grad PyTorch wrt x :", x_torch_grad.grad.shape)
#construit un graphe dynamique 
#pour chaque opération PyTorch crée un nœud dans un graphe 
# il garde en mémoire les valeurs nécessaires pour le backward. 
#  PyTorch remonte ce graphe en utilisant chainrule
#%%


#keops
def gaussianconv_keops(X, Y, S):
    x_i = LazyTensor(X[:, None, :])
    y_j = LazyTensor(Y[None, :, :])
    D_ij = ((x_i - y_j) ** 2).sum(-1)
    K_ij = (- D_ij / (2 * S**2)).exp()
    f = K_ij.sum(dim=1)
    return f.view(X.shape[0], 1)

x_keops_grad = x_torch.clone().requires_grad_(True)

f_keops_grad = gaussianconv_keops(x_keops_grad, y_torch, sigma)
f_keops_grad.sum().backward()

print("Grad KeOps wrt x   :", x_keops_grad.grad.shape)
#il dérive directement la formule mathématique qu’on lui donne
# puis compile automatiquement un kernel optimisé pour calculer le gradient. 
# Du coup, le backward ne stocke jamais toutes les paires (i,j)
# ce qui permet de traiter des très grands jeux de données.

#%%
@jax.jit
def gaussianconv_jax(X, Y, S):
    D_ij = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    K_ij = jnp.exp(-D_ij / (2 * S**2))
    f = jnp.sum(K_ij, axis=1)
    return f.reshape((X.shape[0], 1))

def gaussianconv_jax_scalar(X, Y, sigma):
    f_vec = gaussianconv_jax(X, Y, sigma)   # (N,1)
    return jnp.sum(f_vec)                   # → scalaire pour le backward

# Dérivée ∂f/∂X pour tala convolution
jax_grad_fn = jax.grad(gaussianconv_jax_scalar)

# Gradient wrt X
g_jax = jax_grad_fn(x_jax, y_jax, sigma)

print("Gradient JAX wrt X :", g_jax.shape)
print("dtype du gradient  :", g_jax.dtype)
#regarde toutes les opérations 
#construit un graphe de calcul
#puis génère automatiquement le code du gradient en appliquant la règle de dérivation en chaine. 
# Ensuite, il compile le tout avec XLA

#%%


#  Conversion des gradients en NumPy pour comparaison
grad_torch_np = x_torch_grad.grad.detach().numpy()
grad_jax_np   = np.array(g_jax)

grad_diff_abs = np.max(np.abs(grad_torch_np - grad_jax_np))

eps = 1e-12
grad_diff_rel = np.max(
    np.abs(grad_torch_np - grad_jax_np) / (np.abs(grad_torch_np) + eps)
)

print("Différence ABSOLUE max :", grad_diff_abs)
print("Différence RELATIVE max :", grad_diff_rel)


# %%
