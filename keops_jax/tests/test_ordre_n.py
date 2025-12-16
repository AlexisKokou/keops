#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

# Ajouter le chemin du projet
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from keops_jax.core.jax_interface3 import jax_keops_convolution

print("🧪 TEST DES DÉRIVÉES D'ORDRE N AVEC JAX-KEOPS")
print("=" * 60)

# Données de test
key = jax.random.PRNGKey(42)
M, N, D = 5, 5, 3
X = jax.random.normal(key, (M, D), dtype=jnp.float32)
Y = jax.random.normal(key, (N, D), dtype=jnp.float32)
B = jax.random.normal(key, (N, 1), dtype=jnp.float32)

print(f"Données: M={M}, N={N}, D={D}")
print(f"X: {X.shape}, Y: {Y.shape}, B: {B.shape}")

# Fonction de test
def f(x):
    return jax_keops_convolution("conv_gaussienne", x, Y, B)

def scalar_f(x):
    """Version scalaire pour les tests d'ordre supérieur"""
    return jnp.sum(f(x))

print("\n📊 TESTS DES DIFFÉRENTS ORDRES:")

# Ordre 0 (Forward)
print("\n🔹 Ordre 0 (Forward):")
try:
    result_0 = f(X)
    print(f"   ✅ Shape: {result_0.shape}")
    print(f"   ✅ Valeur: {jnp.mean(result_0):.6f}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Ordre 1 (Gradient)
print("\n🔹 Ordre 1 (Gradient):")
try:
    grad_f = jax.grad(scalar_f)
    result_1 = grad_f(X)
    print(f"   ✅ Shape: {result_1.shape}")
    print(f"   ✅ Norme: {jnp.linalg.norm(result_1):.6f}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Ordre 2 (Hessien - diagonal)
print("\n🔹 Ordre 2 (Hessien):")
try:
    hess_f = jax.hessian(scalar_f)
    result_2 = hess_f(X)
    print(f"   ✅ Shape: {result_2.shape}")
    print(f"   ✅ Trace: {jnp.trace(result_2.reshape(M*D, M*D)):.6f}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Ordre 3 (Dérivée troisième)
print("\n🔹 Ordre 3 (Dérivée 3ème):")
try:
    def third_f(x):
        return jax.grad(jax.grad(scalar_f))(x)[0, 0]  # Un élément du gradient
    
    result_3 = jax.grad(third_f)(X)
    print(f"   ✅ Shape: {result_3.shape}")
    print(f"   ✅ Valeur: {jnp.mean(result_3):.6f}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

print("\n" + "=" * 60)
print("🎯 RÉSUMÉ:")
print("   - Ordre 0: Forward pass")
print("   - Ordre 1: Gradient (dérivée première)")  
print("   - Ordre 2: Hessien (dérivée seconde)")
print("   - Ordre 3: Dérivée troisième")
print("\n✅ Test terminé!")