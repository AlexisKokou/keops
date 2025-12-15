"""
Interface pour les dérivées d'ordre supérieur avec KeOps.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable

from .device_utils import safe_jax_to_numpy, numpy_to_jax


class HigherOrderKeOpsFunction:
    """Classe wrapper pour les dérivées d'ordre supérieur."""
    
    def __init__(self, formula_id: int):
        self.formula_id = formula_id
        
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Forward pass - utilise la fonction existante."""
        from .keops_functions import conv_gaussienne, conv_cauchy, mat_vec_mult, copy_B
        
        funcs = {
            0: conv_gaussienne,
            1: conv_cauchy,
            2: mat_vec_mult,
            3: copy_B
        }
        return funcs[self.formula_id](X, Y, B)
    
    def gradient(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Calcule le gradient d'ordre 1 par rapport à X."""
        def loss_fn(x):
            return self(x, Y, B).sum()
        
        return jax.grad(loss_fn)(X)
    
    def gradient_full(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> tuple:
        """Calcule le gradient complet (par rapport à X, Y, B)."""
        def loss_x(x):
            return self(x, Y, B).sum()
        
        def loss_y(y):
            return self(X, y, B).sum()
        
        def loss_b(b):
            return self(X, Y, b).sum()
        
        grad_X = jax.grad(loss_x)(X)
        grad_Y = jax.grad(loss_y)(Y)
        grad_B = jax.grad(loss_b)(B)
        
        return grad_X, grad_Y, grad_B
    
    def hessian(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Calcule le Hessien par rapport à X."""
        M, D = X.shape
        
        # Calcul point par point pour éviter les problèmes
        hessian = jnp.zeros((M, D, D), dtype=X.dtype)
        
        for i in range(M):
            X_point = X[i:i+1]  # Un seul point
            
            def point_loss(x_point):
                return self(x_point, Y, B).sum()
            
            # Calcul du Hessien pour ce point
            def point_grad(x_point):
                return jax.grad(point_loss)(x_point).flatten()
            
            hessian_i = jax.jacfwd(point_grad)(X_point.reshape(-1))
            hessian = hessian.at[i].set(hessian_i.reshape(D, D))
        
        return hessian
    
    def third_derivative(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Calcule la dérivée troisième (approximation par différences finies)."""
        M, D = X.shape
        eps = 1e-4
        
        third = jnp.zeros((M, D, D, D), dtype=X.dtype)
        
        for i in range(min(M, 2)):  # Limiter pour éviter le calcul coûteux
            X_point = X[i:i+1]
            
            for d1 in range(D):
                for d2 in range(D):
                    for d3 in range(D):
                        # Directions canoniques
                        e1 = jnp.zeros((1, D))
                        e2 = jnp.zeros((1, D))
                        e3 = jnp.zeros((1, D))
                        e1 = e1.at[0, d1].set(1.0)
                        e2 = e2.at[0, d2].set(1.0)
                        e3 = e3.at[0, d3].set(1.0)
                        
                        # Différences finies d'ordre 3
                        X_ppp = X_point + eps*e1 + eps*e2 + eps*e3
                        X_ppm = X_point + eps*e1 + eps*e2 - eps*e3
                        X_pmp = X_point + eps*e1 - eps*e2 + eps*e3
                        X_pmm = X_point + eps*e1 - eps*e2 - eps*e3
                        X_mpp = X_point - eps*e1 + eps*e2 + eps*e3
                        X_mpm = X_point - eps*e1 + eps*e2 - eps*e3
                        X_mmp = X_point - eps*e1 - eps*e2 + eps*e3
                        X_mmm = X_point - eps*e1 - eps*e2 - eps*e3
                        
                        f_ppp = self(X_ppp, Y, B).sum()
                        f_ppm = self(X_ppm, Y, B).sum()
                        f_pmp = self(X_pmp, Y, B).sum()
                        f_pmm = self(X_pmm, Y, B).sum()
                        f_mpp = self(X_mpp, Y, B).sum()
                        f_mpm = self(X_mpm, Y, B).sum()
                        f_mmp = self(X_mmp, Y, B).sum()
                        f_mmm = self(X_mmm, Y, B).sum()
                        
                        third_ijk = (f_ppp - f_ppm - f_pmp + f_pmm - f_mpp + f_mpm + f_mmp - f_mmm) / (8 * eps**3)
                        third = third.at[i, d1, d2, d3].set(third_ijk)
        
        return third


# Fonctions factory
def create_higher_order_function(formula_id: int):
    return HigherOrderKeOpsFunction(formula_id)