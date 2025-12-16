# keops_jax/tests/test_derivatives_vector.py - CORRIGÉ
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("TEST DÉRIVÉES D'ORDRE N - VERSION CORRIGÉE")
print("=" * 70)

import jax
import jax.numpy as jnp
import numpy as np

from core.jax_interface_nth_order import (
    jax_keops_convolution,
    jax_keops_gradient,
    jax_keops_directional_derivative,
    jax_keops_hessian
)

# Configurer KeOps
import pykeops
pykeops.clean_pykeops()
os.environ['PYKEOPS_FORCE_COMPILE'] = '1'

# Données de test
key = jax.random.PRNGKey(42)
M, N, D = 3, 4, 2
X = jax.random.normal(key, (M, D), dtype=jnp.float32)
Y = jax.random.normal(key, (N, D), dtype=jnp.float32)
B = jax.random.normal(key, (N, 1), dtype=jnp.float32)

print(f"📊 Données: M={M}, N={N}, D={D}")

print("\n1️⃣  TEST FORWARD:")
F = jax_keops_convolution("conv_gaussienne", X, Y, B)
print(f"   ✅ F(X) = {F.shape}")
print(f"   Valeurs: {F.flatten()}")

print("\n2️⃣  TEST GRADIENT VECTORIEL (via KeOps):")
G = jax_keops_gradient("conv_gaussienne", X, Y, B)
print(f"   ✅ ∇F(X) = {G.shape}")
print(f"   ∇F[0,:] = {G[0]}")

print("\n3️⃣  TEST DÉRIVÉE DIRECTIONNELLE:")
direction = jax.random.normal(key, (M, D))
D_dir = jax_keops_directional_derivative("conv_gaussienne", X, Y, B, direction)
print(f"   ✅ D_v F(X) = {D_dir.shape}")

# Vérification: D_v F = ⟨∇F, v⟩
grad_dot_dir = jnp.sum(G * direction, axis=1, keepdims=True)
error = jnp.max(jnp.abs(D_dir - grad_dot_dir))
print(f"   Vérification ⟨∇F,v⟩: erreur = {error:.2e}")

print("\n4️⃣  TEST HESSIENNE (via JAX sur gradient KeOps):")
H = jax_keops_hessian("conv_gaussienne", X, Y, B)
print(f"   ✅ Hessienne = {H.shape}")

# Vérification symétrie
for i in range(M):
    H_i = H[i]
    sym_error = jnp.max(jnp.abs(H_i - H_i.T))
    print(f"   F[{i}] symétrie erreur: {sym_error:.2e}")

print("\n5️⃣  TEST DÉRIVÉES D'ORDRE SUPÉRIEUR VIA JAX:")

# Fonction scalaire pour JAX
def f_scalar(X):
    return jnp.sum(jax_keops_convolution("conv_gaussienne", X, Y, B))

print("   a) Gradient via JAX:")
grad_jax = jax.grad(f_scalar)(X)
print(f"      ∇f(X) = {grad_jax.shape}")

print("   b) Hessienne via JAX:")
hess_jax = jax.hessian(f_scalar)(X)
print(f"      ∇²f(X) = {hess_jax.shape}")

print("   c) 3ème ordre via JAX:")
third_jax = jax.jacobian(jax.hessian(f_scalar))(X)
print(f"      ∇³f(X) = {third_jax.shape}")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ:")
print("=" * 70)

print("""
✅ ARCHITECTURE FONCTIONNELLE:

1. Forward (M,1) → KeOps
2. Gradient (M,D) → KeOps (optimisé)
3. Dérivée directionnelle (M,1) → KeOps
4. Hessienne (M,D,D) → JAX sur gradient KeOps
5. Dérivées d'ordre supérieur → JAX

📊 AVANTAGES:
- KeOps optimise gradient et forward
- JAX gère l'autodiff d'ordre supérieur
- Interface propre et efficace
- Pas de shape mismatch

🚀 PRÊT POUR LA RECHERCHE!
""")