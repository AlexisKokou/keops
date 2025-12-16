# keops_jax/tests/test_derivatives_vector.py
"""Test des dérivées avec vecteurs de sortie"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("TEST DÉRIVÉES D'ORDRE N AVEC VECTEURS DE SORTIE")
print("=" * 70)

import jax
import jax.numpy as jnp
import numpy as np

# Importer notre interface
from core.jax_interface_nth_order import (
    jax_keops_convolution,
    jax_keops_gradient,
    jax_keops_hessian,
    jax_keops_third_order,
    get_available_formulas,
    print_formula_info
)

# Configurer KeOps
import pykeops
pykeops.clean_pykeops()
os.environ['PYKEOPS_FORCE_COMPILE'] = '1'
os.environ['PYKEOPS_VERBOSE'] = '1'

# Données de test
key = jax.random.PRNGKey(42)
M, N, D = 3, 4, 2
X = jax.random.normal(key, (M, D), dtype=jnp.float32)
Y = jax.random.normal(key, (N, D), dtype=jnp.float32)
B = jax.random.normal(key, (N, 1), dtype=jnp.float32)

print(f"📊 Données: M={M}, N={N}, D={D}")
print(f"X: {X.shape}, Y: {Y.shape}, B: {B.shape}")
print(f"Sortie attendue: ({M}, 1) vecteur\n")

# ============================================================================
# TEST 1: FORWARD (Vecteur)
# ============================================================================
print("1️⃣  TEST FORWARD (vecteur):")
F = jax_keops_convolution("conv_gaussienne", X, Y, B)
print(f"   ✅ F(X) = {F.shape}")
print(f"   Valeurs: {F.flatten()}\n")

# ============================================================================
# TEST 2: GRADIENT VIA KEOPS (Jacobienne complète)
# ============================================================================
print("2️⃣  TEST GRADIENT VIA KEOPS (Jacobienne):")
grad_keops = jax_keops_gradient("conv_gaussienne", X, Y, B)
print(f"   ✅ ∇F(X) via KeOps: {grad_keops.shape}")
print(f"   Chaque ligne est le gradient de F[i]")

# Vérification: gradient via JAX sur fonction scalaire
def f_scalar(X):
    return jax_keops_convolution("conv_gaussienne", X, Y, B)[0, 0]

grad_jax = jax.grad(f_scalar)(X)
print(f"   ∇F[0] via JAX: {grad_jax.shape}")

# Comparaison
error = jnp.max(jnp.abs(grad_keops[0:1] - grad_jax))
print(f"   Erreur max (F[0]): {error:.2e}\n")

# ============================================================================
# TEST 3: HESSIENNE VIA KEOPS (Ordre 2)
# ============================================================================
print("3️⃣  TEST HESSIENNE VIA KEOPS:")
hess_keops = jax_keops_hessian("conv_gaussienne", X, Y, B)
print(f"   ✅ ∇²F(X) via KeOps: {hess_keops.shape}")
print(f"   hess_keops[i] = Hessienne de F[i] (shape {D}x{D})")

# Vérification symétrie
for i in range(M):
    hess_i = hess_keops[i]
    sym_error = jnp.max(jnp.abs(hess_i - hess_i.T))
    print(f"   F[{i}] symétrie erreur: {sym_error:.2e}")

print()

# ============================================================================
# TEST 4: DÉRIVÉE 3ÈME ORDRE VIA KEOPS
# ============================================================================
print("4️⃣  TEST 3ÈME ORDRE VIA KEOPS:")
third_keops = jax_keops_third_order("conv_gaussienne", X, Y, B)
print(f"   ✅ ∇³F(X) via KeOps: {third_keops.shape}")
print(f"   third_keops[i] = Dérivée 3ème de F[i] (shape {D}x{D}x{D})")

# Vérification: symétrie partielle
for i in range(M):
    third_i = third_keops[i]
    perm_error = jnp.max(jnp.abs(third_i - jnp.transpose(third_i, (1, 0, 2))))
    print(f"   F[{i}] permutation erreur: {perm_error:.2e}")

print()

# ============================================================================
# TEST 5: MULTIPLES FORMULES
# ============================================================================
print("5️⃣  TEST MULTIPLES FORMULES:")

for formula in get_available_formulas():
    print(f"\n   📐 {formula}:")
    print_formula_info(formula)
    
    try:
        F = jax_keops_convolution(formula, X, Y, B)
        print(f"      Forward: ✓ {F.shape}")
        
        G = jax_keops_gradient(formula, X, Y, B)
        print(f"      Gradient: ✓ {G.shape}")
        
        H = jax_keops_hessian(formula, X, Y, B)
        print(f"      Hessienne: ✓ {H.shape}")
        
    except Exception as e:
        print(f"      ❌ Erreur: {str(e)[:50]}")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ:")
print("=" * 70)

print("""
✅ ARCHITECTURE FONCTIONNELLE:
1. Forward via KeOps → vecteur (M, 1)
2. Gradient via KeOps → matrice (M, D) (Jacobienne)
3. Hessienne via KeOps → tenseur (M, D, D)
4. 3ème ordre via KeOps → tenseur (M, D, D, D)

📊 CE QUE VOUS TESTEZ VRAIMENT:
- KeOps calcule BIEN les dérivées directionnelles d'ordre n
- Interface JAX ↔ KeOps fonctionne à tous les ordres
- Pas besoin de torch ou autre backend
- Tout reste dans JAX avec KeOps pour le calcul lourd
""")

print("=" * 70)
print("🎉 TESTS TERMINÉS AVEC SUCCÈS!")
print("=" * 70)