# core/jax_interface2.py
import jax
import jax.numpy as jnp
from .formulas import FORMULAS, FORMULA_STRINGS
from .keops_executor_derivate import keops_forward, keops_grad1, keops_grad2

# ---------- wrappers JAX -> pure_callback (non-différentiable) ----------

def jax_keops_forward(formula_name, X, Y, B):
    """
    Non-diff wrapper: forward via KeOps (pure_callback).
    formula_name : string key from FORMULAS
    X, Y, B : jax arrays (float32)
    returns jax array (N,1)
    """
    fid = FORMULAS[formula_name]
    M = X.shape[0]
    out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)

    def cb(fid_, X_np, Y_np, B_np):
        return keops_forward(int(fid_), X_np, Y_np, B_np, FORMULA_STRINGS)

    fid_j = jnp.asarray(int(fid), dtype=jnp.int32)
    return jax.pure_callback(cb, out_shape, fid_j, X, Y, B)


def jax_keops_grad1(formula_name, X, Y, B, G):
    """
    Non-diff wrapper: compute first derivative via KeOps.
    G : upstream gradient (N,1)
    returns dX (N,D)
    """
    fid = FORMULAS[formula_name]
    M, D = X.shape
    out_shape = jax.ShapeDtypeStruct((M, D), X.dtype)

    def cb(fid_, X_np, Y_np, B_np, G_np):
        return keops_grad1(int(fid_), X_np, Y_np, B_np, G_np, FORMULA_STRINGS)

    fid_j = jnp.asarray(int(fid), dtype=jnp.int32)
    return jax.pure_callback(cb, out_shape, fid_j, X, Y, B, G)


def jax_keops_grad2(formula_name, X, Y, B, G1, G2):
    """
    Non-diff wrapper: second-order derivative primitive via KeOps.
    G1, G2 : cotangents (N,) or (N,1)
    """
    fid = FORMULAS[formula_name]
    M, D = X.shape
    out_shape = jax.ShapeDtypeStruct((M, D), X.dtype)

    def cb(fid_, X_np, Y_np, B_np, G1_np, G2_np):
        return keops_grad2(int(fid_), X_np, Y_np, B_np, G1_np, G2_np, FORMULA_STRINGS)

    fid_j = jnp.asarray(int(fid), dtype=jnp.int32)
    return jax.pure_callback(cb, out_shape, fid_j, X, Y, B, G1, G2)
