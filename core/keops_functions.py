# keops_jax/core/keops_functions.py
"""
Interface KeOps-JAX - SOLUTION CORRIGÉE
Backend KeOps + Syntaxe JAX pure avec support VJP
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Callable

from .keops_executor import keops_forward
from .keops_autodiff import DERIVATIVE_GENERATOR
from .device_utils import safe_jax_to_numpy, numpy_to_jax


def create_keops_function(formula_id: int):
    """Crée une fonction KeOps avec forward et VJP."""
    
    # --------------------------------------------------------
    # 1. FONCTIONS PURE CALLBACK
    # --------------------------------------------------------
    def forward_callback(X, Y, B):
        """Callback pure pour forward (KeOps Genred)."""
        X_np = safe_jax_to_numpy(X)
        Y_np = safe_jax_to_numpy(Y)
        B_np = safe_jax_to_numpy(B)
        return keops_forward(formula_id, X_np, Y_np, B_np)
    
    # --------------------------------------------------------
    # 2. FONCTION PRINCIPALE AVEC CUSTOM VJP
    # --------------------------------------------------------
    @jax.custom_vjp
    def keops_func(X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Fonction KeOps principale - syntaxe JAX pure."""
        result_shape = jax.ShapeDtypeStruct((X.shape[0], 1), X.dtype)
        return jax.pure_callback(
            forward_callback,
            result_shape,
            X, Y, B,
            vmap_method='sequential'
        )
    
    # --------------------------------------------------------
    # 3. RÈGLES VJP
    # --------------------------------------------------------
    def keops_func_fwd(X, Y, B):
        return keops_func(X, Y, B), (X, Y, B)
    
    def keops_func_bwd(residuals, cotangent):
        X, Y, B = residuals
        
        def gradient_callback(x, y, b, ct):
            """Callback pour calculer tous les gradients."""
            X_np = safe_jax_to_numpy(x)
            Y_np = safe_jax_to_numpy(y)
            B_np = safe_jax_to_numpy(b)
            ct_np = safe_jax_to_numpy(ct)
            
            # Utiliser DERIVATIVE_GENERATOR pour les gradients
            grad_X = DERIVATIVE_GENERATOR.compute_derivative(
                formula_id, ('X',), X_np, Y_np, B_np, (ct_np,)
            )
            
            grad_Y_i = DERIVATIVE_GENERATOR.compute_derivative(
                formula_id, ('Y',), X_np, Y_np, B_np, (ct_np,)
            )
            
            grad_B_i = DERIVATIVE_GENERATOR.compute_derivative(
                formula_id, ('B',), X_np, Y_np, B_np, (ct_np,)
            )
            
            # Convertir grad_Y de i à j
            M, D = X_np.shape
            N = Y_np.shape[0]
            
            grad_Y = np.zeros((N, D), dtype=Y_np.dtype)
            for j in range(N):
                i = j % M
                if grad_Y_i.ndim == 2 and grad_Y_i.shape[1] == D:
                    grad_Y[j] = grad_Y_i[i]
                else:
                    grad_Y[j] = grad_Y_i[i, 0] if grad_Y_i.ndim == 2 else grad_Y_i[i]
            
            # Convertir grad_B de i à j
            grad_B = np.zeros((N, 1), dtype=B_np.dtype)
            for j in range(N):
                i = j % M
                grad_B[j] = grad_B_i[i, 0] if grad_B_i.ndim == 2 else grad_B_i[i]
            
            return grad_X, grad_Y, grad_B
        
        grad_X_shape = jax.ShapeDtypeStruct(X.shape, X.dtype)
        grad_Y_shape = jax.ShapeDtypeStruct(Y.shape, Y.dtype)
        grad_B_shape = jax.ShapeDtypeStruct(B.shape, B.dtype)
        
        grad_X, grad_Y, grad_B = jax.pure_callback(
            lambda x, y, b, ct: gradient_callback(x, y, b, ct),
            (grad_X_shape, grad_Y_shape, grad_B_shape),
            X, Y, B, cotangent,
            vmap_method='sequential'
        )
        
        return (grad_X, grad_Y, grad_B)
    
    keops_func.defvjp(keops_func_fwd, keops_func_bwd)
    
    return keops_func


# ------------------------------------------------------------
# FONCTIONS EXPORTÉES
# ------------------------------------------------------------
conv_gaussienne = create_keops_function(0)
conv_cauchy = create_keops_function(1)
mat_vec_mult = create_keops_function(2)
copy_B = create_keops_function(3)