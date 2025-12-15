# keops_jax/core/keops_jit.py
"""
Version optimisée pour JIT des fonctions KeOps.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

from .device_utils import safe_jax_to_numpy, numpy_to_jax
from .keops_executor import keops_forward
from .keops_autodiff import DERIVATIVE_GENERATOR


def create_jittable_keops_function(formula_id: int):
    """
    Crée une fonction KeOps compatible JIT.
    Utilise jax.pure_callback pour intégrer KeOps dans JAX.
    """
    
    # Définir la fonction pure (sans side effects) pour le callback
    def keops_pure_forward(X, Y, B):
        X_np = safe_jax_to_numpy(X)
        Y_np = safe_jax_to_numpy(Y)
        B_np = safe_jax_to_numpy(B)
        return keops_forward(formula_id, X_np, Y_np, B_np)
    
    def keops_pure_backward(variable: str, X, Y, B, cotangent):
        """Calcule le gradient pour une variable spécifique."""
        X_np = safe_jax_to_numpy(X)
        Y_np = safe_jax_to_numpy(Y)
        B_np = safe_jax_to_numpy(B)
        cotangent_np = safe_jax_to_numpy(cotangent)
        
        return DERIVATIVE_GENERATOR.compute_derivative(
            formula_id, (variable,), X_np, Y_np, B_np, (cotangent_np,)
        )
    
    # Fonction wrapper avec jax.pure_callback
    @jax.custom_vjp
    def keops_func(X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        # Pendant JIT, utiliser pure_callback
        result_shape = jax.ShapeDtypeStruct((X.shape[0], 1), X.dtype)
        
        result = jax.pure_callback(
            keops_pure_forward,
            result_shape,
            X, Y, B,
            vmap_method='sequential'
        )
        return result
    
    # Forward pass
    def keops_func_fwd(X, Y, B):
        primal_out = keops_func(X, Y, B)
        return primal_out, (X, Y, B)
    
    # Backward pass
    def keops_func_bwd(residuals, cotangent):
        X, Y, B = residuals
        
        # Shapes pour les callbacks
        grad_X_shape = jax.ShapeDtypeStruct(X.shape, X.dtype)
        grad_Y_shape = jax.ShapeDtypeStruct(Y.shape, Y.dtype)
        grad_B_shape = jax.ShapeDtypeStruct(B.shape, B.dtype)
        
        # Calculer chaque gradient avec pure_callback
        grad_X = jax.pure_callback(
            lambda *args: keops_pure_backward('X', *args),
            grad_X_shape,
            X, Y, B, cotangent,
            vmap_method='sequential'
        )
        
        grad_Y = jax.pure_callback(
            lambda *args: keops_pure_backward('Y', *args),
            grad_Y_shape,
            X, Y, B, cotangent,
            vmap_method='sequential'
        )
        
        grad_B = jax.pure_callback(
            lambda *args: keops_pure_backward('B', *args),
            grad_B_shape,
            X, Y, B, cotangent,
            vmap_method='sequential'
        )
        
        return (grad_X, grad_Y, grad_B)
    
    keops_func.defvjp(keops_func_fwd, keops_func_bwd)
    
    return keops_func


# Créer les fonctions JIT-friendly
conv_gaussienne_jit = create_jittable_keops_function(0)
conv_cauchy_jit = create_jittable_keops_function(1)
mat_vec_mult_jit = create_jittable_keops_function(2)
copy_B_jit = create_jittable_keops_function(3)