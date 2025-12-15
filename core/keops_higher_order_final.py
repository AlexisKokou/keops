"""
Interface pour les dérivées d'ordre supérieur avec VRAI backend KeOps.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pykeops.numpy import Genred
from typing import Tuple, List

from .device_utils import safe_jax_to_numpy, numpy_to_jax
from .formulas import FORMULA_STRINGS


class TrueKeOpsHigherOrder:
    """Vraies dérivées d'ordre supérieur avec KeOps."""
    
    def __init__(self, formula_id: int):
        self.formula_id = formula_id
        self._kernels = {}
        
    def _build_derivative_formula(self, variables: Tuple[str, ...]) -> Tuple[str, List[str]]:
        """Construit la formule KeOps pour les dérivées d'ordre n."""
        base_formula = FORMULA_STRINGS[self.formula_id]
        
        current = base_formula
        aliases = []
        
        # Les aliases seront complétés avec la dimension réelle plus tard
        for i, var in enumerate(variables):
            eta = f"eta{i}"
            current = f"Grad({current}, {var}, {eta})"
            
        return current, aliases
    
    def get_derivative_kernel(self, variables: Tuple[str, ...], D: int) -> Genred:
        """Obtient le noyau KeOps pour une dérivée d'ordre n."""
        cache_key = (self.formula_id, tuple(variables), D)
        
        if cache_key not in self._kernels:
            formula, _ = self._build_derivative_formula(variables)
            
            # Construire les aliases dynamiquement
            aliases = [f"X = Vi({D})", f"Y = Vj({D})", "B = Vj(1)"]
            
            for i, var in enumerate(variables):
                if var == 'X':
                    aliases.append(f"eta{i} = Vi({D})")
                else:  # 'Y' ou 'B'
                    if var == 'Y':
                        aliases.append(f"eta{i} = Vj({D})")
                    else:  # 'B'
                        aliases.append(f"eta{i} = Vj(1)")
            
            kernel = Genred(formula, aliases, reduction_op="Sum", axis=1)
            self._kernels[cache_key] = kernel
        
        return self._kernels[cache_key]
    
    def compute_derivative(self, variables: Tuple[str, ...], 
                          X: np.ndarray, Y: np.ndarray, B: np.ndarray,
                          etas: Tuple[np.ndarray, ...]) -> np.ndarray:
        """Calcule une dérivée d'ordre n avec KeOps."""
        D = X.shape[1]
        kernel = self.get_derivative_kernel(variables, D)
        
        args = [X, Y, B] + list(etas)
        result = kernel(*args)
        
        # Ajuster la shape si nécessaire
        if variables[-1] == 'X' and result.shape == (X.shape[0], 1):
            result = np.repeat(result, D, axis=1)
        
        return result


class HigherOrderKeOpsFunction:
    """Fonction avec vraies dérivées d'ordre supérieur KeOps."""
    
    def __init__(self, formula_id: int):
        self.formula_id = formula_id
        self.derivative_calculator = TrueKeOpsHigherOrder(formula_id)
        self._forward_func = None
        
    def _get_forward_func(self):
        """Charge la fonction forward."""
        if self._forward_func is None:
            from .keops_functions import conv_gaussienne, conv_cauchy, mat_vec_mult, copy_B
            funcs = {
                0: conv_gaussienne,
                1: conv_cauchy,
                2: mat_vec_mult,
                3: copy_B
            }
            self._forward_func = funcs[self.formula_id]
        return self._forward_func
    
    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Forward pass."""
        return self._get_forward_func()(X, Y, B)
    
    def gradient(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Calcule le gradient avec VRAI KeOps Grad()."""
        M, D = X.shape
        
        def gradient_callback(x, y, b):
            X_np = safe_jax_to_numpy(x)
            Y_np = safe_jax_to_numpy(y)
            B_np = safe_jax_to_numpy(b)
            
            # Pour chaque direction canonique
            grad = np.zeros((M, D), dtype=X_np.dtype)
            
            for d in range(D):
                eta = np.zeros((M, D), dtype=X_np.dtype)
                eta[:, d] = 1.0
                
                result = self.derivative_calculator.compute_derivative(
                    ('X',), X_np, Y_np, B_np, (eta,)
                )
                grad[:, d] = result.flatten() if result.shape == (M, 1) else result[:, d]
            
            return grad
        
        grad_shape = jax.ShapeDtypeStruct((M, D), X.dtype)
        return jax.pure_callback(
            gradient_callback,
            grad_shape,
            X, Y, B,
            vmap_method='sequential'
        )
    
    def hessian(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Calcule le Hessien avec VRAI KeOps Grad(Grad())."""
        M, D = X.shape
        
        def hessian_callback(x, y, b):
            X_np = safe_jax_to_numpy(x)
            Y_np = safe_jax_to_numpy(y)
            B_np = safe_jax_to_numpy(b)
            
            hessian = np.zeros((M, D, D), dtype=X_np.dtype)
            
            for i in range(D):
                for j in range(D):
                    eta1 = np.zeros((M, D), dtype=X_np.dtype)
                    eta2 = np.zeros((M, D), dtype=X_np.dtype)
                    eta1[:, i] = 1.0
                    eta2[:, j] = 1.0
                    
                    # Grad(Grad(f, X, eta1), X, eta2)
                    result = self.derivative_calculator.compute_derivative(
                        ('X', 'X'), X_np, Y_np, B_np, (eta1, eta2)
                    )
                    
                    # Résultat est un scalaire par point
                    if result.shape == (M, 1):
                        hessian[:, i, j] = result.flatten()
                    elif result.shape == (M, D):
                        # Prendre le produit scalaire avec eta2
                        hessian[:, i, j] = np.sum(result * eta2, axis=1)
            
            return hessian
        
        hess_shape = jax.ShapeDtypeStruct((M, D, D), X.dtype)
        return jax.pure_callback(
            hessian_callback,
            hess_shape,
            X, Y, B,
            vmap_method='sequential'
        )
    
    def third_derivative(self, X: jnp.ndarray, Y: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        """Calcule la dérivée troisième avec VRAI KeOps Grad(Grad(Grad()))."""
        M, D = X.shape
        
        def third_derivative_callback(x, y, b):
            X_np = safe_jax_to_numpy(x)
            Y_np = safe_jax_to_numpy(y)
            B_np = safe_jax_to_numpy(b)
            
            third = np.zeros((M, D, D, D), dtype=X_np.dtype)
            
            for i in range(D):
                for j in range(D):
                    for k in range(D):
                        eta1 = np.zeros((M, D), dtype=X_np.dtype)
                        eta2 = np.zeros((M, D), dtype=X_np.dtype)
                        eta3 = np.zeros((M, D), dtype=X_np.dtype)
                        eta1[:, i] = 1.0
                        eta2[:, j] = 1.0
                        eta3[:, k] = 1.0
                        
                        # Grad(Grad(Grad(f, X, eta1), X, eta2), X, eta3)
                        result = self.derivative_calculator.compute_derivative(
                            ('X', 'X', 'X'), X_np, Y_np, B_np, (eta1, eta2, eta3)
                        )
                        
                        if result.shape == (M, 1):
                            third[:, i, j, k] = result.flatten()
            
            return third
        
        third_shape = jax.ShapeDtypeStruct((M, D, D, D), X.dtype)
        return jax.pure_callback(
            third_derivative_callback,
            third_shape,
            X, Y, B,
            vmap_method='sequential'
        )


# Fonctions factory
def create_higher_order_function(formula_id: int):
    return HigherOrderKeOpsFunction(formula_id)


# Instances pré-créées
higher_order_gaussian = create_higher_order_function(0)
higher_order_cauchy = create_higher_order_function(1)
higher_order_mat_vec_mult = create_higher_order_function(2)
higher_order_copy_B = create_higher_order_function(3)