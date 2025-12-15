# keops_jax/core/keops_autodiff.py
"""
Générateur de dérivées KeOps.
"""
import numpy as np
from pykeops.numpy import Genred
from typing import Tuple, Dict, List
from .formulas import FORMULA_STRINGS

class KeOpsDerivativeGenerator:
    def __init__(self):
        self._formula_cache = {}
        
    def _generate_derivative_formula(self, base_formula: str, variables: Tuple[str, ...]) -> str:
        current = base_formula
        for i, var in enumerate(variables):
            eta = f"eta{i}"
            current = f"Grad({current}, {var}, {eta})"
        return current
    
    def _get_aliases(self, D: int, derivatives: Tuple[str, ...]) -> List[str]:
        aliases = [f"X = Vi({D})", f"Y = Vj({D})", "B = Vj(1)"]
        for i, var in enumerate(derivatives):
            if var == 'X':
                aliases.append(f"eta{i} = Vi(1)")
            else:  # Y ou B
                aliases.append(f"eta{i} = Vj(1)")
        return aliases
    
    def get_derivative_kernel(self, formula_id: int, derivatives: Tuple[str, ...], D: int) -> Genred:
        cache_key = (formula_id, derivatives, D)
        if cache_key not in self._formula_cache:
            base_formula = FORMULA_STRINGS[formula_id]
            derivative_formula = self._generate_derivative_formula(base_formula, derivatives)
            aliases = self._get_aliases(D, derivatives)
            kernel = Genred(derivative_formula, aliases, reduction_op="Sum", axis=1)
            self._formula_cache[cache_key] = kernel
        return self._formula_cache[cache_key]
    
    def compute_derivative(self, formula_id: int, derivatives: Tuple[str, ...],
                          X: np.ndarray, Y: np.ndarray, B: np.ndarray,
                          etas: Tuple[np.ndarray, ...]) -> np.ndarray:
        """Calcule une dérivée."""
        D = X.shape[1]
        M = X.shape[0]
        N = Y.shape[0]
        
        kernel = self.get_derivative_kernel(formula_id, derivatives, D)
        args = [X, Y, B] + list(etas)
        result = kernel(*args)
        
        last_var = derivatives[-1]
        
        if last_var == 'X':
            if result.shape == (M, 1):
                result = np.repeat(result, D, axis=1)
            return result
            
        elif last_var == 'Y':
            if result.shape == (M, 1):
                result_on_j = np.zeros((N, D), dtype=result.dtype)
                for j in range(N):
                    result_on_j[j] = result[j % M]
                return result_on_j
            elif result.shape == (M, D):
                result_on_j = np.zeros((N, D), dtype=result.dtype)
                for j in range(N):
                    result_on_j[j] = result[j % M]
                return result_on_j
            else:
                return result
                
        else:  # 'B'
            if result.shape == (M, 1):
                result_on_j = np.zeros((N, 1), dtype=result.dtype)
                for j in range(N):
                    result_on_j[j] = result[j % M]
                return result_on_j
            else:
                return result

DERIVATIVE_GENERATOR = KeOpsDerivativeGenerator()