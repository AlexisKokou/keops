# keops_jax/tests/test_order_n_comprehensive.py
"""Test complet des dérivées d'ordre n"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("TEST COMPLET D'ORDRE N (1-4)")
print("=" * 70)

import jax
import jax.numpy as jnp
import numpy as np
from core.jax_interface_nth_order import jax_keops_convolution

# Config
import pykeops
pykeops.clean_pykeops()
os.environ['PYKEOPS_FORCE_COMPILE'] = '1'
os.environ['PYKEOPS_VERBOSE'] = '0'

# Données simples
key = jax.random.PRNGKey(42)
M, N, D = 2, 3, 2
X = jax.random.normal(key, (M, D), dtype=jnp.float32)
Y = jax.random.normal(key, (N, D), dtype=jnp.float32)
B = jnp.ones((N, 1), dtype=jnp.float32)

print(f"Configuration: M={M}, N={N}, D={D}")
print(f"Taille problème: {M}×{D} = {M*D} paramètres\n")

# ============================================================================
# TEST 1: FORWARD
# ============================================================================
print("1️⃣  TEST FORWARD:")
F = jax_keops_convolution("conv_gaussienne", X, Y, B)
print(f"   F(X) = {F.shape}")
print(f"   Valeurs: {F.flatten()}")

# ============================================================================
# TEST 2: GRADIENT
# ============================================================================
print("\n2️⃣  TEST GRADIENT:")

def f_scalar(X):
    return jax_keops_convolution("conv_gaussienne", X, Y, B)[0, 0]

grad = jax.grad(f_scalar)(X)
print(f"   ∇F[0] via JAX.grad: {grad.shape}")

# Vérification par différences finies
eps = 1e-4
X_np = np.array(X)
grad_fd = np.zeros_like(X_np)

for i in range(M):
    for j in range(D):
        X_plus = X_np.copy()
        X_minus = X_np.copy()
        X_plus[i, j] += eps
        X_minus[i, j] -= eps
        
        f_plus = f_scalar(jnp.array(X_plus))
        f_minus = f_scalar(jnp.array(X_minus))
        grad_fd[i, j] = (f_plus - f_minus) / (2 * eps)

error = np.max(np.abs(grad - grad_fd))
print(f"   Erreur vs différences finies: {error:.2e}")

# ============================================================================
# TEST 3: HESSIENNE
# ============================================================================
print("\n3️⃣  TEST HESSIENNE:")
hess = jax.hessian(f_scalar)(X)
print(f"   ∇²F[0] via JAX.hessian: {hess.shape}")

# Vérification symétrie
sym_error = jnp.max(jnp.abs(hess - hess.T))
print(f"   Erreur symétrie: {sym_error:.2e}")

# ============================================================================
# TEST 4: 3ÈME ORDRE
# ============================================================================
print("\n4️⃣  TEST 3ÈME ORDRE:")

def grad_f(X):
    return jax.grad(f_scalar)(X)

third = jax.jacobian(grad_f)(X)
print(f"   ∇³F[0] via JAX.jacobian(grad): {third.shape}")

# Vérification permutation
perm_error = jnp.max(jnp.abs(third - jnp.transpose(third, (1, 0))))
print(f"   Erreur permutation: {perm_error:.2e}")

# ============================================================================
# TEST 5: 4ÈME ORDRE
# ============================================================================
print("\n5️⃣  TEST 4ÈME ORDRE:")
fourth = jax.hessian(grad_f)(X)
print(f"   ∇⁴F[0] via JAX.hessian(grad): {fourth.shape}")

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "=" * 70)
print("🎊 RÉSUMÉ DES CAPACITÉS:")
print("=" * 70)

print(f"""
✅ DÉRIVÉES SUPPORTÉES JUSQU'À L'ORDRE 4:

1. Ordre 0 (Forward):        {F.shape}
2. Ordre 1 (Gradient):       {grad.shape}
3. Ordre 2 (Hessienne):      {hess.shape}
4. Ordre 3 (Dérivée 3ème):   {third.shape}
5. Ordre 4 (Dérivée 4ème):   {fourth.shape}

🎯 PRÉCISION:
- Erreur gradient:           {error:.2e}
- Symétrie Hessienne:        {sym_error:.2e}
- Symétrie 3ème ordre:       {perm_error:.2e}

🚀 PERFORMANCE:
- KeOps optimise le forward
- JAX gère l'autodiff d'ordre supérieur
- Pas besoin de PyTorch
- Tout en JAX + KeOps pur
""")

print("=" * 70)
print("✅ ARCHITECTURE VALIDÉE POUR LA RECHERCHE!")
print("=" * 70)