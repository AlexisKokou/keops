# keops_executor_derivate3.py - Version CORRECTE

import numpy as np
from pykeops.numpy import Genred

_KERNEL_CACHE = {}

def make_nth_order_kernel(formula_string, D, order):
    cache_key = (formula_string, D, order)
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]
    
    if order == 0:
        aliases = [
            f"X = Vi({D})",
            f"Y = Vj({D})",
            "B = Vj(1)",
        ]
        kernel = Genred(formula_string, aliases, reduction_op="Sum", axis=1)
    else:
        current_formula = formula_string
        aliases = [
            f"X = Vi({D})",
            f"Y = Vj({D})",
            "B = Vj(1)",
        ]
        
        for i in range(1, order + 1):
            current_formula = f"Grad({current_formula}, X, V{i})"
            aliases.append(f"V{i} = Vi({D})")
        
        kernel = Genred(current_formula, aliases, reduction_op="Sum", axis=1)
    
    _KERNEL_CACHE[cache_key] = kernel
    return kernel

def keops_nth_order(formula_id, X_np, Y_np, B_np, *direction_vectors, FORMULA_STRINGS):
    X_np = np.asarray(X_np, dtype=np.float32)
    Y_np = np.asarray(Y_np, dtype=np.float32)
    B_np = np.asarray(B_np, dtype=np.float32)
    
    direction_arrays = [np.asarray(v, dtype=np.float32) for v in direction_vectors]
    
    N, D = X_np.shape
    order = len(direction_arrays)
    
    formula = FORMULA_STRINGS[formula_id]
    kernel = make_nth_order_kernel(formula, D, order)
    
    result = kernel(X_np, Y_np, B_np, *direction_arrays)
    return result

def keops_forward(formula_id, X_np, Y_np, B_np, FORMULA_STRINGS):
    return keops_nth_order(formula_id, X_np, Y_np, B_np, FORMULA_STRINGS=FORMULA_STRINGS)

def keops_backward(formula_id, X_np, Y_np, B_np, G_np, FORMULA_STRINGS):
    return keops_nth_order(formula_id, X_np, Y_np, B_np, G_np, FORMULA_STRINGS=FORMULA_STRINGS)

def keops_hessian(formula_id, X_np, Y_np, B_np, G_np, H_np, FORMULA_STRINGS):
    return keops_nth_order(formula_id, X_np, Y_np, B_np, G_np, H_np, FORMULA_STRINGS=FORMULA_STRINGS)
