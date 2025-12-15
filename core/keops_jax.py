# keops_jax/core/keops_jax.py
"""
Interface principale KeOps pour JAX.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from functools import partial

from .device_utils import jax_to_numpy, numpy_to_jax
from .keops_executor import keops_forward
from .keops_autodiff import DERIVATIVE_GENERATOR


class KeOpsJaxKernel:
    """Wrapper JAX pour un noyau KeOps avec autodiff."""
    
    def __init__(self, formula_id: int):
        self.formula_id = formula_id
    
    # Définissons la fonction forward comme une méthode régulière
    def forward(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Forward pass avec conversion automatique."""
        # Conversion JAX -> NumPy
        X_np = jax_to_numpy(X)
        Y_np = jax_to_numpy(Y)
        B_np = jax_to_numpy(B)
        
        # Exécution KeOps
        result_np = keops_forward(self.formula_id, X_np, Y_np, B_np)
        
        # Conversion NumPy -> JAX
        return numpy_to_jax(result_np)
    
    # Nous allons créer le custom_vjp wrapper séparément
    def _make_custom_vjp(self):
        """Crée la fonction avec custom_vjp."""
        
        @jax.custom_vjp
        def kernel_func(X, Y, B):
            return self.forward(X, Y, B)
        
        def kernel_func_fwd(X, Y, B):
            primal_out = kernel_func(X, Y, B)
            return primal_out, (X, Y, B)
        
        def kernel_func_bwd(residuals, cotangent):
            X, Y, B = residuals
            cotangent_np = jax_to_numpy(cotangent)
            
            # Calcul des gradients pour chaque variable d'entrée
            X_np = jax_to_numpy(X)
            Y_np = jax_to_numpy(Y)
            B_np = jax_to_numpy(B)
            
            # Gradient par rapport à X
            grad_X = DERIVATIVE_GENERATOR.compute_derivative(
                self.formula_id, ('X',), X_np, Y_np, B_np, (cotangent_np,)
            )
            
            # Gradient par rapport à Y
            grad_Y = DERIVATIVE_GENERATOR.compute_derivative(
                self.formula_id, ('Y',), X_np, Y_np, B_np, (cotangent_np,)
            )
            
            # Gradient par rapport à B  
            grad_B = DERIVATIVE_GENERATOR.compute_derivative(
                self.formula_id, ('B',), X_np, Y_np, B_np, (cotangent_np,)
            )
            
            return (numpy_to_jax(grad_X), numpy_to_jax(grad_Y), numpy_to_jax(grad_B))
        
        kernel_func.defvjp(kernel_func_fwd, kernel_func_bwd)
        return kernel_func
    
    # Pour permettre la syntaxe d'appel: kernel(X, Y, B)
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        kernel_func = self._make_custom_vjp()
        return kernel_func(X, Y, B)


# Fonctions factory pour créer des noyaux avec autodiff
def gaussian_kernel():
    return KeOpsJaxKernel(formula_id=0)

def cauchy_kernel():
    return KeOpsJaxKernel(formula_id=1)

def dot_product_kernel():
    return KeOpsJaxKernel(formula_id=2)

def copy_kernel():
    return KeOpsJaxKernel(formula_id=3)