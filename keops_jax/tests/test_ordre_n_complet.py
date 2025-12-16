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

from keops_jax.core.jax_interface_ordre_n import jax_keops_convolution, jax_keops_nth_derivative

print("🧪 TEST COMPLET DES DÉRIVÉES D'ORDRE N")
print("=" * 50)

# Configuration
key = jax.random.PRNGKey(42)
M, N, D = 3, 3, 2
X = jax.random.normal(key, (M, D), dtype=jnp.float32)
Y = jax.random.normal(key, (N, D), dtype=jnp.float32)
B = jax.random.normal(key, (N, 1), dtype=jnp.float32)

print(f"Données: M={M}, N={N}, D={D}")

# Tests progressifs
def test_order(order, description):
    print(f"\n🔹 {description} (ordre {order}):")
    try:
        if order == 0:
            # Forward
            result = jax_keops_convolution("conv_gaussienne", X, Y, B)
            print(f"   ✅ Forward: {result.shape} | Valeur: {jnp.mean(result):.6f}")
            
        elif order == 1:
            # Gradient via JAX autodiff
            def f_scalar(x):
                return jnp.sum(jax_keops_convolution("conv_gaussienne", x, Y, B))
            
            grad_result = jax.grad(f_scalar)(X)
            print(f"   ✅ Gradient (autodiff): {grad_result.shape} | Norme: {jnp.linalg.norm(grad_result):.6f}")
            
            # Gradient via dérivée directe (avec vecteur direction)
            direction = jnp.ones_like(X)
            direct_result = jax_keops_nth_derivative("conv_gaussienne", 1, X, Y, B, direction)
            print(f"   ✅ Gradient (direct): {direct_result.shape} | Valeur: {jnp.mean(direct_result):.6f}")
            
        elif order == 2:
            # Hessien via JAX
            def f_scalar(x):
                return jnp.sum(jax_keops_convolution("conv_gaussienne", x, Y, B))
            
            hess_result = jax.hessian(f_scalar)(X)
            print(f"   ✅ Hessien (autodiff): {hess_result.shape} | Trace: {jnp.trace(hess_result.reshape(-1, M*D)):.6f}")
            
            # Dérivée seconde directe
            dir1 = jnp.ones_like(X)
            dir2 = jnp.ones_like(X) * 0.5
            direct_result = jax_keops_nth_derivative("conv_gaussienne", 2, X, Y, B, dir1, dir2)
            print(f"   ✅ Dérivée 2 (direct): {direct_result.shape} | Valeur: {jnp.mean(direct_result):.6f}")
            
        else:
            # Ordres supérieurs (direct seulement)
            directions = [jnp.ones_like(X) * (0.8**i) for i in range(order)]
            direct_result = jax_keops_nth_derivative("conv_gaussienne", order, X, Y, B, *directions)
            print(f"   ✅ Dérivée {order} (direct): {direct_result.shape} | Valeur: {jnp.mean(direct_result):.6f}")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {type(e).__name__}: {e}")
        return False

# Tests progressifs
success_count = 0
total_tests = 5

success_count += test_order(0, "Forward")
success_count += test_order(1, "Gradient") 
success_count += test_order(2, "Hessien")
success_count += test_order(3, "Dérivée 3ème")
success_count += test_order(4, "Dérivée 4ème")

print(f"\n" + "=" * 50)
print(f"🎯 RÉSUMÉ: {success_count}/{total_tests} tests réussis")

if success_count == total_tests:
    print("🎉 PARFAIT! Toutes les dérivées d'ordre n fonctionnent!")
elif success_count >= 3:
    print("✅ BON! Les ordres principaux (0, 1, 2) fonctionnent.")
else:
    print("⚠️ Des problèmes détectés dans les ordres de base.")

print("\n💡 UTILISATION:")
print("  • jax_keops_convolution(formula, X, Y, B)  # Forward + autodiff")
print("  • jax_keops_nth_derivative(formula, n, X, Y, B, *dirs)  # Dérivée ordre n")