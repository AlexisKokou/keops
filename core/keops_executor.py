# keops_jax/core/keops_executor.py
"""
Interface KeOps en NumPy pur pour JAX.
"""
import numpy as np
from pykeops.numpy import Genred
from .formulas import FORMULA_STRINGS

# Cache global pour éviter la recompilation
_KERNEL_CACHE = {}

def make_kernel(formula: str, aliases: list) -> Genred:
    """
    Crée un noyau KeOps avec cache.
    """
    key = (formula, tuple(aliases))
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = Genred(
            formula, aliases, reduction_op="Sum", axis=1
        )
    return _KERNEL_CACHE[key]

def keops_forward(formula_id: int, X_np: np.ndarray, 
                  Y_np: np.ndarray, B_np: np.ndarray) -> np.ndarray:
    """
    Forward pass avec KeOps.
    """
    M, D = X_np.shape
    N = Y_np.shape[0]
    
    # Validation
    if Y_np.shape[1] != D:
        raise ValueError(f"D dimensions mismatch: X has {D}, Y has {Y_np.shape[1]}")
    
    if B_np.shape != (N, 1):
        raise ValueError(f"B must have shape ({N}, 1), got {B_np.shape}")
    
    # Formule et aliases
    formula = FORMULA_STRINGS[formula_id]
    
    # Special case for copy_B formula
    if formula_id == 3:  # copy_B
        # Just sum B over j dimension for each i
        aliases = [
            f"X = Vi({D})",  # X is not used but required for shape
            f"Y = Vj({D})",  # Y is not used but required for shape  
            "B = Vj(1)"
        ]
        formula = "B"  # Simply copy B
    else:
        aliases = [
            f"X = Vi({D})",
            f"Y = Vj({D})",
            "B = Vj(1)"
        ]
    
    # Exécution
    kernel = make_kernel(formula, aliases)
    result = kernel(X_np, Y_np, B_np)
    
    if result.shape != (M, 1):
        raise RuntimeError(f"Unexpected output shape: {result.shape} != ({M}, 1)")
    
    return result