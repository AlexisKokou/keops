import jax
import jax.numpy as jnp
from .formulas import FORMULAS
from .formulas import FORMULA_STRINGS
from .keops_executor import keops_forward, keops_backward


def jax_keops_convolution_impl(formula_id, X, Y, B):
    
    M = X.shape[0]
    out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)

    def fw_callback(formula_id_, X_, Y_, B_):
        # formula_id_ est un jnp.ndarray, on le convertit en int Python
        return keops_forward(int(formula_id_), X_, Y_, B_, FORMULA_STRINGS)

    # Convertit formula_id (int Python) en tableau JAX
    formula_id_jax = jnp.asarray(formula_id, dtype=jnp.int32)
    
    # Tous les arguments sont maintenant des jnp.ndarray
    return jax.pure_callback(fw_callback, out_shape, formula_id_jax, X, Y, B)


# ---------------------------------------------------------------------
# Fonctions VJP (Forward et Backward)

def fwd(formula_id, X, Y, B):
    output = jax_keops_convolution_impl(formula_id, X, Y, B)
    # Les résidus contiennent l'ID de la formule (int)
    return output, (formula_id, X, Y, B)


def bwd(formula_id, res, G):
    # formula_id (int) est passé en premier car il est non-différentiable
    _, X, Y, B = res 
    
    # formula_id est déjà un int, pas besoin de le redériver
    
    M, D = X.shape
    dx_shape = jax.ShapeDtypeStruct((M, D), X.dtype)

    def bw_callback(formula_id_, X_, Y_, B_, G_):
        return keops_backward(int(formula_id_), X_, Y_, B_, G_, FORMULA_STRINGS)

    # Convertit formula_id (int Python) en tableau JAX
    formula_id_jax = jnp.asarray(formula_id, dtype=jnp.int32)
    
    dX = jax.pure_callback(bw_callback, dx_shape, formula_id_jax, X, Y, B, G)

    return dX, None, None



# On définit la fonction VJP pour l'implémentation de base
jax_keops_convolution_vjp = jax.custom_vjp(jax_keops_convolution_impl, nondiff_argnums=(0,)) 
jax_keops_convolution_vjp.defvjp(fwd, bwd)


# FONCTION PUBLIQUE (WRAPPER)

# Fonction wrapper publique qui accepte le nom de la formule (string)
# Elle trouve l'ID et appelle la fonction JAX VJP avec l'ID (int)
def jax_keops_convolution(formula_name, X, Y, B):
    formula_id = FORMULAS[formula_name]
    # Appelle la fonction VJP corrigée. formula_id est l'argument statique.
    return jax_keops_convolution_vjp(formula_id, X, Y, B)