# keops_jax/core/jax_interface_nth_order.py
"""Interface JAX pour KeOps avec support des dérivées d'ordre n"""

import jax
import jax.numpy as jnp
import numpy as np
from .formulas import FORMULAS, FORMULA_STRINGS
from .keops_executor_nth_order import keops_nth_order
from .device_utils import jax_to_numpy, numpy_to_jax

# ============================================================================
# CALLBACK AVEC SUPPORT ORDRE N
# ============================================================================

def keops_callback_nth_order(formula_id, X, Y, B, *direction_vectors):
    """Callback qui appelle KeOps pour l'ordre n"""
    M, D = X.shape
    order = len(direction_vectors)
    
    # Shape de sortie : (M, 1) pour tous les ordres
    out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)
    
    def callback_impl(formula_id_np, X_np, Y_np, B_np, *dir_vecs_np):
        return keops_nth_order(
            int(formula_id_np), X_np, Y_np, B_np, *dir_vecs_np,
            FORMULA_STRINGS=FORMULA_STRINGS
        )
    
    formula_id_np = np.int32(formula_id)
    X_np = jax_to_numpy(X)
    Y_np = jax_to_numpy(Y)
    B_np = jax_to_numpy(B)
    dir_vecs_np = [jax_to_numpy(v) for v in direction_vectors]
    
    return jax.pure_callback(
        callback_impl,
        out_shape,
        formula_id_np, X_np, Y_np, B_np, *dir_vecs_np,
        vmap_method='sequential'
    )

# ============================================================================
# FONCTION VECTORIELLE AVEC CUSTOM_JVP
# ============================================================================

_VECTOR_FUNCTION_CACHE = {}

def make_vector_keops_function(formula_name):
    """Crée une fonction vectorielle qui utilise KeOps pour les dérivées"""
    if formula_name in _VECTOR_FUNCTION_CACHE:
        return _VECTOR_FUNCTION_CACHE[formula_name]
    
    formula_id = FORMULAS[formula_name]
    
    @jax.custom_jvp
    def vector_func(X, Y, B):
        # Forward (ordre 0)
        return keops_callback_nth_order(formula_id, X, Y, B)
    
    @vector_func.defjvp
    def vector_func_jvp(primals, tangents):
        X, Y, B = primals
        X_dot, Y_dot, B_dot = tangents
        
        # Forward
        primal_out = vector_func(X, Y, B)
        
        # JVP : dérivée directionnelle via KeOps
        if X_dot is not None:
            tangent_out = keops_callback_nth_order(formula_id, X, Y, B, X_dot)
        else:
            tangent_out = jnp.zeros_like(primal_out)
        
        return primal_out, tangent_out
    
    _VECTOR_FUNCTION_CACHE[formula_name] = vector_func
    return vector_func

# ============================================================================
# FONCTIONS D'ORDRE SUPÉRIEUR DIRECTES
# ============================================================================

def jax_keops_gradient(formula_name, X, Y, B):
    """Gradient complet (Jacobienne) via KeOps"""
    formula_id = FORMULAS[formula_name]
    M, D = X.shape
    
    gradients = []
    for d in range(D):
        direction = jnp.zeros_like(X)
        direction = direction.at[:, d].set(1.0)
        dir_grad = keops_callback_nth_order(formula_id, X, Y, B, direction)
        gradients.append(dir_grad)
    
    return jnp.concatenate(gradients, axis=1)

def jax_keops_hessian(formula_name, X, Y, B):
    """Hessienne complète via KeOps"""
    formula_id = FORMULAS[formula_name]
    M, D = X.shape
    
    hessian_blocks = []
    for d1 in range(D):
        row_blocks = []
        for d2 in range(D):
            direction1 = jnp.zeros_like(X).at[:, d1].set(1.0)
            direction2 = jnp.zeros_like(X).at[:, d2].set(1.0)
            hess_block = keops_callback_nth_order(formula_id, X, Y, B, direction1, direction2)
            row_blocks.append(hess_block)
        row = jnp.concatenate(row_blocks, axis=1)
        hessian_blocks.append(row)
    
    return jnp.stack(hessian_blocks, axis=1)

def jax_keops_third_order(formula_name, X, Y, B):
    """Dérivée troisième via KeOps"""
    formula_id = FORMULAS[formula_name]
    M, D = X.shape
    
    third_order = []
    for d1 in range(D):
        for d2 in range(D):
            for d3 in range(D):
                directions = [
                    jnp.zeros_like(X).at[:, d1].set(1.0),
                    jnp.zeros_like(X).at[:, d2].set(1.0),
                    jnp.zeros_like(X).at[:, d3].set(1.0)
                ]
                third_block = keops_callback_nth_order(formula_id, X, Y, B, *directions)
                third_order.append(third_block)
    
    third_array = jnp.stack(third_order, axis=1)
    return third_array.reshape(M, D, D, D)

def jax_keops_nth_order_directional(formula_name, X, Y, B, *directions):
    """Dérivée directionnelle d'ordre n"""
    formula_id = FORMULAS[formula_name]
    return keops_callback_nth_order(formula_id, X, Y, B, *directions)

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def jax_keops_convolution(formula_name, X, Y, B):
    """Interface principale - retourne un vecteur (M, 1)"""
    return make_vector_keops_function(formula_name)(X, Y, B)

# ============================================================================
# FONCTION SCALAIRE POUR JAX.GRAD
# ============================================================================

_SCALAR_FUNCTION_CACHE = {}

def make_scalar_keops_function(formula_name):
    """Crée une fonction scalaire compatible avec jax.grad"""
    if formula_name in _SCALAR_FUNCTION_CACHE:
        return _SCALAR_FUNCTION_CACHE[formula_name]
    
    vector_func = make_vector_keops_function(formula_name)
    formula_id = FORMULAS[formula_name]
    
    @jax.custom_jvp
    def scalar_func(X, Y, B):
        return jnp.sum(vector_func(X, Y, B))
    
    @scalar_func.defjvp
    def scalar_func_jvp(primals, tangents):
        X, Y, B = primals
        X_dot, Y_dot, B_dot = tangents
        
        primal_out = scalar_func(X, Y, B)
        
        if X_dot is not None:
            dir_deriv = keops_callback_nth_order(formula_id, X, Y, B, X_dot)
            tangent_out = jnp.sum(dir_deriv)
        else:
            tangent_out = jnp.array(0.0, dtype=X.dtype)
        
        return primal_out, tangent_out
    
    _SCALAR_FUNCTION_CACHE[formula_name] = scalar_func
    return scalar_func

def jax_keops_convolution_scalar(formula_name, X, Y, B):
    """Version scalaire pour jax.grad"""
    return make_scalar_keops_function(formula_name)(X, Y, B)

# ============================================================================
# UTILITAIRES
# ============================================================================

def get_available_formulas():
    """Retourne la liste des formules disponibles"""
    return list(FORMULAS.keys())

def print_formula_info(formula_name):
    """Affiche des informations sur une formule"""
    if formula_name not in FORMULAS:
        print(f"❌ Formule '{formula_name}' non trouvée")
        return
    
    formula_id = FORMULAS[formula_name]
    formula_string = FORMULA_STRINGS[formula_id]
    
    print(f"📐 Formule: {formula_name}")
    print(f"   ID: {formula_id}")
    print(f"   Expression KeOps: {formula_string}")
    print(f"   Description: ", end="")
    
    if formula_id == 0:
        print("Convolution avec noyau gaussien")
    elif formula_id == 1:
        print("Convolution avec noyau de Cauchy")
    elif formula_id == 2:
        print("Multiplication matrice-vecteur")
    elif formula_id == 3:
        print("Copie du vecteur B")