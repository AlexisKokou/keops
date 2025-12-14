# jax_interface3_simple.py - Version avec custom_jvp seulement

import jax
import jax.numpy as jnp
import numpy as np
from .formulas import FORMULAS, FORMULA_STRINGS
from .keops_executor_derivate3 import keops_nth_order

# Cache des fonctions
_FUNCTION_CACHE = {}

def get_keops_function_by_order(formula_name, order):
    cache_key = (formula_name, order)
    if cache_key in _FUNCTION_CACHE:
        return _FUNCTION_CACHE[cache_key]
    
    formula_id = FORMULAS[formula_name]
    
    def keops_func(X, Y, B, *direction_vectors):
        X_np = np.array(X, dtype=np.float32)
        Y_np = np.array(Y, dtype=np.float32)
        B_np = np.array(B, dtype=np.float32)
        dir_vecs_np = [np.array(v, dtype=np.float32) for v in direction_vectors]
        
        result_np = keops_nth_order(
            formula_id, X_np, Y_np, B_np, *dir_vecs_np,
            FORMULA_STRINGS=FORMULA_STRINGS
        )
        
        return jnp.array(result_np)
    
    _FUNCTION_CACHE[cache_key] = keops_func
    return keops_func

# Fonction principale avec custom_jvp
def make_keops_function(formula_name):
    func0 = get_keops_function_by_order(formula_name, 0)
    
    @jax.custom_jvp
    def keops_func(X, Y, B):
        return func0(X, Y, B)
    
    @keops_func.defjvp
    def keops_func_jvp(primals, tangents):
        X, Y, B = primals
        X_dot, Y_dot, B_dot = tangents
        
        # Output primal
        primal_out = keops_func(X, Y, B)
        
        # Calcul de la dérivée directionnelle
        if X_dot is not None:
            func1 = get_keops_function_by_order(formula_name, 1)
            tangent_out = func1(X, Y, B, X_dot)
        else:
            tangent_out = jnp.zeros_like(primal_out)
        
        return primal_out, tangent_out
    
    return keops_func

def jax_keops_convolution(formula_name, X, Y, B):
    if formula_name not in _FUNCTION_CACHE:
        _FUNCTION_CACHE[formula_name] = make_keops_function(formula_name)
    return _FUNCTION_CACHE[formula_name](X, Y, B)