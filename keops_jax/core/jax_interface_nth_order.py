# keops_jax/core/jax_interface_nth_order.py
"""Interface JAX pour KeOps - CORRIGÉ"""

import jax
import jax.numpy as jnp
import numpy as np
from formulas import FORMULAS, FORMULA_STRINGS
from keops_executor_nth_order import keops_nth_order, keops_gradient_vector, keops_directional_derivative
from device_utils import jax_to_numpy, numpy_to_jax

# ============================================================================
# CALLBACKS CORRIGÉS
# ============================================================================

def keops_callback_gradient_vector(formula_id, X, Y, B):
    """Callback pour le gradient vectoriel (M, D)"""
    M, D = X.shape
    out_shape = jax.ShapeDtypeStruct((M, D), X.dtype)
    
    def callback_impl(formula_id_np, X_np, Y_np, B_np):
        return keops_gradient_vector(
            int(formula_id_np), X_np, Y_np, B_np, FORMULA_STRINGS
        )
    
    # Utilisation de pure_callback avec les arrays JAX directement
    return jax.pure_callback(
        callback_impl,
        out_shape,
        np.int32(formula_id), X, Y, B,
        vmap_method='sequential'
    )

def keops_callback_directional(formula_id, X, Y, B, direction):
    """Callback pour la dérivée directionnelle (M, 1)"""
    M, D = X.shape
    out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)
    
    def callback_impl(formula_id_np, X_np, Y_np, B_np, direction_np):
        return keops_directional_derivative(
            int(formula_id_np), X_np, Y_np, B_np, direction_np, FORMULA_STRINGS
        )
    
    # Utilisation de pure_callback avec les arrays JAX directement
    return jax.pure_callback(
        callback_impl,
        out_shape,
        np.int32(formula_id), X, Y, B, direction,
        vmap_method='sequential'
    )

# ============================================================================
# FONCTIONS PRINCIPALES CORRIGÉES
# ============================================================================

def jax_keops_gradient(formula_name, X, Y, B):
    """Gradient complet (M, D) via KeOps - CORRIGÉ"""
    formula_id = FORMULAS[formula_name]
    return keops_callback_gradient_vector(formula_id, X, Y, B)

def jax_keops_directional_derivative(formula_name, X, Y, B, direction):
    """Dérivée directionnelle (M, 1) via KeOps"""
    formula_id = FORMULAS[formula_name]
    return keops_callback_directional(formula_id, X, Y, B, direction)

def jax_keops_second_order(formula_name, X, Y, B, direction1, direction2):
    """Dérivée seconde directionnelle via KeOps"""
    formula_id = FORMULAS[formula_name]
    return keops_callback_second_order(formula_id, X, Y, B, direction1, direction2)

def jax_keops_hessian(formula_name, X, Y, B):
    """Hessienne complète (M, D, D) via KeOps - APPROCHE ALTERNATIVE"""
    formula_id = FORMULAS[formula_name]
    M, D = X.shape
    
    # Calculer le gradient vectoriel
    grad_func = lambda X: keops_callback_gradient_vector(formula_id, X, Y, B)
    
    # Calculer la Jacobienne du gradient = Hessienne
    hessian = jax.jacobian(grad_func)(X)
    
    # Reformatage: hessian a shape (M, D, M, D), on veut (M, D, D) pour chaque point
    # Note: Ceci calcule la Hessienne complète (toutes les interactions entre points)
    # Pour une Hessienne par point (indépendante), on prend la diagonale
    hessian_diag = hessian[jnp.arange(M), :, jnp.arange(M), :]
    
    return hessian_diag

def jax_keops_hessian_directional(formula_name, X, Y, B, direction1, direction2):
    """Dérivée directionnelle d'ordre 2 (M, 1) via KeOps"""
    formula_id = FORMULAS[formula_name]
    M, D = X.shape
    
    # Utiliser KeOps pour la dérivée directionnelle d'ordre 2
    # Construction de la formule Grad(Grad(f,X,V1),X,V2)
    out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)
    
    def callback_impl(formula_id_np, X_np, Y_np, B_np, dir1_np, dir2_np):
        return keops_nth_order(
            int(formula_id_np), X_np, Y_np, B_np, FORMULA_STRINGS, dir1_np, dir2_np
        )
    
    formula_id_np = np.int32(formula_id)
    X_np = jax_to_numpy(X)
    Y_np = jax_to_numpy(Y)
    B_np = jax_to_numpy(B)
    dir1_np = jax_to_numpy(direction1)
    dir2_np = jax_to_numpy(direction2)
    
    return jax.pure_callback(
        callback_impl,
        out_shape,
        formula_id_np, X_np, Y_np, B_np, dir1_np, dir2_np,
        vmap_method='sequential'
    )

# ============================================================================
# INTERFACE PRINCIPALE (inchangée)
# ============================================================================

def make_vector_keops_function(formula_name):
    """Crée une fonction vectorielle avec custom_jvp"""
    formula_id = FORMULAS[formula_name]
    
    @jax.custom_jvp
    def vector_func(X, Y, B):
        # Forward (ordre 0)
        M, D = X.shape
        out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)
        
        def callback_impl(formula_id_np, X_np, Y_np, B_np):
            return keops_nth_order(
                int(formula_id_np), X_np, Y_np, B_np, FORMULA_STRINGS
            )
        
        formula_id_np = np.int32(formula_id)
        X_np = jax_to_numpy(X)
        Y_np = jax_to_numpy(Y)
        B_np = jax_to_numpy(B)
        
        return jax.pure_callback(
            callback_impl,
            out_shape,
            formula_id_np, X_np, Y_np, B_np,
            vmap_method='sequential'
        )
    
    @vector_func.defjvp
    def vector_func_jvp(primals, tangents):
        X, Y, B = primals
        X_dot, Y_dot, B_dot = tangents
        
        # Forward
        primal_out = vector_func(X, Y, B)
        
        # JVP : dérivée directionnelle via KeOps
        if X_dot is not None:
            tangent_out = keops_callback_directional(formula_id, X, Y, B, X_dot)
        else:
            tangent_out = jnp.zeros_like(primal_out)
        
        return primal_out, tangent_out
    
    return vector_func

_VECTOR_FUNCTION_CACHE = {}

def jax_keops_convolution(formula_name, X, Y, B):
    """Interface principale - retourne un vecteur (M, 1)"""
    if formula_name not in _VECTOR_FUNCTION_CACHE:
        _VECTOR_FUNCTION_CACHE[formula_name] = make_vector_keops_function(formula_name)
    
    func = _VECTOR_FUNCTION_CACHE[formula_name]
    return func(X, Y, B)

# ============================================================================
# TEST SIMPLIFIÉ
# ============================================================================

def test_interface():
    """Test rapide de l'interface corrigée"""
    import pykeops
    pykeops.clean_pykeops()
    
    key = jax.random.PRNGKey(42)
    M, N, D = 3, 4, 2
    X = jax.random.normal(key, (M, D))
    Y = jax.random.normal(key, (N, D))
    B = jnp.ones((N, 1))
    
    print("Test forward:")
    F = jax_keops_convolution("conv_gaussienne", X, Y, B)
    print(f"  F shape: {F.shape} (attendu: ({M}, 1))")
    
    print("\nTest gradient vectoriel:")
    G = jax_keops_gradient("conv_gaussienne", X, Y, B)
    print(f"  ∇F shape: {G.shape} (attendu: ({M}, {D}))")
    
    print("\nTest dérivée directionnelle:")
    direction = jax.random.normal(key, (M, D))
    D_dir = jax_keops_directional_derivative("conv_gaussienne", X, Y, B, direction)
    print(f"  D_v F shape: {D_dir.shape} (attendu: ({M}, 1))")
    
    # Vérification: D_v F ≈ ⟨∇F, v⟩
    grad_v = jnp.sum(G * direction, axis=1, keepdims=True)
    error = jnp.max(jnp.abs(D_dir - grad_v))
    print(f"  Erreur D_v F vs ⟨∇F,v⟩: {error:.2e}")
    
    return F, G, D_dir

if __name__ == "__main__":
    test_interface()