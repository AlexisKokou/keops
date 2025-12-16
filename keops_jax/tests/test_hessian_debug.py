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

from keops_jax.core.jax_interface_hessian import jax_keops_convolution

print("🧪 TEST SPÉCIFIQUE DES HESSIENNES JAX-KEOPS")
print("=" * 50)

# Configuration simple pour debug
key = jax.random.PRNGKey(42)
M, N, D = 2, 2, 2  # Très petit pour debug
X = jax.random.normal(key, (M, D), dtype=jnp.float32) * 0.1  # Petites valeurs
Y = jax.random.normal(key, (N, D), dtype=jnp.float32) * 0.1
B = jax.random.normal(key, (N, 1), dtype=jnp.float32) * 0.1

print(f"Données: M={M}, N={N}, D={D}")
print(f"X:\n{X}")
print(f"Y:\n{Y}")
print(f"B:\n{B}")

# Fonction de test
def f_scalar(x):
    """Fonction scalaire pour tester hessienne"""
    result = jax_keops_convolution("conv_gaussienne", x, Y, B)
    return jnp.sum(result)

print("\n📊 TESTS PROGRESSIFS:")

# Test 1: Forward
print("\n🔹 Test 1 - Forward:")
try:
    result_0 = jax_keops_convolution("conv_gaussienne", X, Y, B)
    print(f"   ✅ Forward: {result_0.shape}")
    print(f"   Valeurs:\n{result_0}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 2: Gradient
print("\n🔹 Test 2 - Gradient:")
try:
    grad_fn = jax.grad(f_scalar)
    result_1 = grad_fn(X)
    print(f"   ✅ Gradient: {result_1.shape}")
    print(f"   Valeurs:\n{result_1}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 3: JVP (produit jacobien-vecteur)
print("\n🔹 Test 3 - JVP:")
try:
    direction = jnp.ones_like(X)
    
    def jvp_test(x, v):
        return jax.jvp(f_scalar, (x,), (v,))
    
    primal, tangent = jvp_test(X, direction)
    print(f"   ✅ JVP primal: {primal}")
    print(f"   ✅ JVP tangent: {tangent}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 4: Hessienne (objectif principal)
print("\n🔹 Test 4 - HESSIENNE:")
try:
    hess_fn = jax.hessian(f_scalar)
    result_hess = hess_fn(X)
    print(f"   ✅ Hessienne: {result_hess.shape}")
    print(f"   Trace: {jnp.trace(result_hess.reshape(-1, M*D)):.6f}")
    print(f"   Valeurs (extrait):\n{result_hess[0, 0, :, :]}")
except Exception as e:
    print(f"   ❌ Erreur: {e}")

# Test 5: Comparaison avec différences finies
print("\n🔹 Test 5 - Validation par différences finies:")
try:
    def finite_diff_hessian(f, x, eps=1e-4):
        """Hessien par différences finies"""
        grad_f = jax.grad(f)
        
        def finite_diff_grad(x, eps):
            M, D = x.shape
            hess = np.zeros((M, D, M, D))
            
            for i in range(M):
                for j in range(D):
                    x_plus = x.at[i, j].add(eps)
                    x_minus = x.at[i, j].add(-eps)
                    
                    grad_plus = grad_f(x_plus)
                    grad_minus = grad_f(x_minus)
                    
                    hess[i, j, :, :] = (grad_plus - grad_minus) / (2 * eps)
            
            return hess
        
        return finite_diff_grad(x, eps)
    
    # Hessien par différences finies
    hess_fd = finite_diff_hessian(f_scalar, X, eps=1e-4)
    
    # Comparaison
    if 'result_hess' in locals():
        error = jnp.linalg.norm(result_hess - hess_fd) / jnp.linalg.norm(hess_fd + 1e-10)
        print(f"   ✅ Erreur relative vs diff. finies: {error:.6f}")
        
        if error < 1e-2:
            print("   🎉 EXCELLENT! Hessienne JAX-KeOps validée!")
        elif error < 1e-1:
            print("   ✅ BON! Hessienne acceptable.")
        else:
            print("   ⚠️ Hessienne imprécise.")
    else:
        print("   ⚠️ Pas de hessienne JAX-KeOps à comparer")
        
except Exception as e:
    print(f"   ❌ Erreur: {e}")

print("\n" + "=" * 50)
print("🎯 DIAGNOSTIC HESSIENNE JAX-KEOPS:")
print("   Si tous les tests passent, la hessienne fonctionne!")
print("   Sinon, ajustements nécessaires dans l'interface.")