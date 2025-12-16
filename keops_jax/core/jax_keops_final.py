# jax_keops_final.py - Interface JAX-KeOps finale pour dérivées d'ordre n

import jax
import jax.numpy as jnp
import numpy as np

# ============================================================================
# BACKEND KEOPS PUR
# ============================================================================

def keops_forward_numpy(formula_id, X_np, Y_np, B_np):
    """Forward KeOps pur en NumPy"""
    from pykeops.numpy import Genred
    
    formulas = [
        "Exp(-SqDist(X,Y)) * B",      # 0: conv_gaussienne
        "IntInv(IntCst(1) + SqDist(X,Y)) * B",  # 1: conv_cauchy 
        "(X | Y) * B",                # 2: mat_vec_mult
        "B"                           # 3: copy_B
    ]
    
    formula_names = {
        "conv_gaussienne": 0,
        "conv_cauchy": 1, 
        "mat_vec_mult": 2,
        "copy_B": 3
    }
    
    M, D = X_np.shape
    formula = formulas[formula_id]
    aliases = [f"X = Vi({D})", f"Y = Vj({D})", "B = Vj(1)"]
    
    kernel = Genred(formula, aliases, reduction_op='Sum', axis=1)
    result = kernel(X_np.astype(np.float32), Y_np.astype(np.float32), B_np.astype(np.float32))
    return result

def keops_gradient_numpy(formula_id, X_np, Y_np, B_np, direction_np):
    """Gradient KeOps via Grad en NumPy"""
    from pykeops.numpy import Genred
    
    formulas = [
        "Exp(-SqDist(X,Y)) * B",      # 0: conv_gaussienne
        "IntInv(IntCst(1) + SqDist(X,Y)) * B",  # 1: conv_cauchy 
        "(X | Y) * B",                # 2: mat_vec_mult
        "B"                           # 3: copy_B
    ]
    
    M, D = X_np.shape
    formula = formulas[formula_id]
    
    # Construction du Grad
    grad_formula = f"Grad({formula}, X, V)"
    aliases = [f"X = Vi({D})", f"Y = Vj({D})", "B = Vj(1)", f"V = Vi({D})"]
    
    kernel = Genred(grad_formula, aliases, reduction_op='Sum', axis=1)
    result = kernel(X_np.astype(np.float32), Y_np.astype(np.float32), 
                   B_np.astype(np.float32), direction_np.astype(np.float32))
    return result

# ============================================================================
# INTERFACE JAX AVEC CUSTOM_JVP SEULEMENT
# ============================================================================

@jax.custom_jvp
def _jax_keops_vectorial_impl(formula_id, X, Y, B):
    """Fonction vectorielle KeOps (M, 1) avec autodiff KeOps complet - implémentation interne"""
    
    M = X.shape[0]
    
    # Forward via KeOps
    result_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)
    
    def forward_callback(formula_id_val, X_val, Y_val, B_val):
        # Conversion explicite en NumPy
        X_np = np.asarray(X_val, dtype=np.float32)
        Y_np = np.asarray(Y_val, dtype=np.float32)  
        B_np = np.asarray(B_val, dtype=np.float32)
        return keops_forward_numpy(int(formula_id_val), X_np, Y_np, B_np)
    
    return jax.pure_callback(
        forward_callback, 
        result_shape,
        formula_id, X, Y, B,
        vmap_method='sequential'
    )

# Définition du JVP pour les dérivées d'ordre supérieur
@_jax_keops_vectorial_impl.defjvp
def _jax_keops_vectorial_impl_jvp(primals, tangents):
    """JVP pour dérivées d'ordre supérieur"""
    formula_id, X, Y, B = primals
    d_formula_id, d_X, d_Y, d_B = tangents
    
    # Forward
    primal_out = _jax_keops_vectorial_impl(formula_id, X, Y, B)
    
    # JVP via KeOps Grad pour la direction X 
    if d_X is not None:
        M, D = X.shape
        
        def jvp_X_callback(formula_id_val, X_val, Y_val, B_val, dX_val):
            # Conversion explicite en NumPy
            X_np = np.asarray(X_val, dtype=np.float32)
            Y_np = np.asarray(Y_val, dtype=np.float32)
            B_np = np.asarray(B_val, dtype=np.float32)
            dX_np = np.asarray(dX_val, dtype=np.float32)
            return keops_gradient_numpy(int(formula_id_val), X_np, Y_np, B_np, dX_np)
        
        tangent_out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)
        tangent_out = jax.pure_callback(
            jvp_X_callback,
            tangent_out_shape,
            formula_id, X, Y, B, d_X,
            vmap_method='sequential'
        )
    else:
        tangent_out = jnp.zeros_like(primal_out)
    
    return primal_out, tangent_out

# ============================================================================
# FONCTIONS PRINCIPALES
# ============================================================================

def jax_keops_vectorial(formula_name, X, Y, B):
    """Interface utilisateur pour fonction vectorielle"""
    # Mapping nom -> id
    formula_names = {
        "conv_gaussienne": 0,
        "conv_cauchy": 1, 
        "mat_vec_mult": 2,
        "copy_B": 3
    }
    
    formula_id = formula_names[formula_name]
    return _jax_keops_vectorial_impl(formula_id, X, Y, B)

def jax_keops_scalar(formula_name, X, Y, B):
    """Version scalaire pour dérivées d'ordre supérieur"""
    vectorial_result = jax_keops_vectorial(formula_name, X, Y, B)
    return jnp.sum(vectorial_result)

# Alias pour compatibilité
jax_keops_convolution = jax_keops_vectorial

# ============================================================================
# FORMULAS POUR RÉFÉRENCE
# ============================================================================

FORMULA_INFO = {
    "conv_gaussienne": {"id": 0, "formula": "Exp(-SqDist(X,Y)) * B"},
    "conv_cauchy": {"id": 1, "formula": "IntInv(IntCst(1) + SqDist(X,Y)) * B"},
    "mat_vec_mult": {"id": 2, "formula": "(X | Y) * B"},
    "copy_B": {"id": 3, "formula": "B"}
}