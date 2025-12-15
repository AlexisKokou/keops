# keops_jax/core/keops_jvp.py
"""
Implémentation optimisée des VJP pour KeOps.
"""
import numpy as np
from pykeops.numpy import Genred
from typing import Tuple, List
from .formulas import FORMULA_STRINGS


class KeOpsVJPGenerator:
    """Génère des noyaux KeOps pour le VJP."""
    
    def __init__(self):
        self._kernel_cache = {}
    
    def _get_vjp_kernel(self, formula_id: int, var: str, D: int) -> Genred:
        """Crée un noyau KeOps pour le VJP d'une variable spécifique."""
        cache_key = (formula_id, var, D)
        
        if cache_key not in self._kernel_cache:
            base_formula = FORMULA_STRINGS[formula_id]
            
            # Formule pour le gradient par rapport à la variable
            if var == 'X':
                formula = f"Grad({base_formula}, X, v)"
                aliases = [
                    f"X = Vi({D})",
                    f"Y = Vj({D})",
                    "B = Vj(1)",
                    "v = Vi(1)"
                ]
                reduction = "Sum"
                
            elif var == 'Y':
                formula = f"Grad({base_formula}, Y, v)"
                aliases = [
                    f"X = Vi({D})",
                    f"Y = Vj({D})",
                    "B = Vj(1)",
                    "v = Vi(1)"
                ]
                reduction = "Sum"
                
            else:  # 'B'
                formula = f"Grad({base_formula}, B, v)"
                aliases = [
                    f"X = Vi({D})",
                    f"Y = Vj({D})",
                    "B = Vj(1)",
                    "v = Vi(1)"
                ]
                reduction = "Sum"
            
            kernel = Genred(formula, aliases, reduction_op=reduction, axis=1)
            self._kernel_cache[cache_key] = kernel
        
        return self._kernel_cache[cache_key]
    
    def compute_vjp(self, formula_id: int, var: str,
                    X: np.ndarray, Y: np.ndarray, B: np.ndarray,
                    cotangent: np.ndarray) -> np.ndarray:
        """Calcule le VJP pour une variable spécifique."""
        D = X.shape[1]
        kernel = self._get_vjp_kernel(formula_id, var, D)
        
        if var == 'X':
            result = kernel(X, Y, B, cotangent)
            if result.shape == (X.shape[0], 1):
                result = np.repeat(result, D, axis=1)
            return result
            
        elif var == 'Y':
            result = kernel(X, Y, B, cotangent)
            if result.shape == (X.shape[0], 1):
                result = np.repeat(result, D, axis=1)
            # Pour Y, on doit transférer de i à j
            N = Y.shape[0]
            M = X.shape[0]
            result_on_j = np.zeros((N, D), dtype=result.dtype)
            for j in range(N):
                i = j % M
                result_on_j[j] = result[i]
            return result_on_j
            
        else:  # 'B'
            result = kernel(X, Y, B, cotangent)
            # Pour B, on doit transférer de i à j
            N = Y.shape[0]
            M = X.shape[0]
            result_on_j = np.zeros((N, 1), dtype=result.dtype)
            for j in range(N):
                i = j % M
                result_on_j[j] = result[i]
            return result_on_j


# Exportez les deux pour la compatibilité
VJP_GENERATOR = KeOpsVJPGenerator()
JVP_GENERATOR = KeOpsVJPGenerator()  # Alias pour compatibilité