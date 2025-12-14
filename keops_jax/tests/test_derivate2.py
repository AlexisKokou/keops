# keops_jax/tests/test_derivate_fixed.py
"""
Test complet: forward / grad / grad(grad) en utilisant la pile KeOps-JAX.
Assure-toi d'avoir pykeops installé (et torch).
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

from keops_jax.core.jax_interface3 import jax_keops_convolution
from keops_jax.core.formulas import FORMULAS, FORMULA_STRINGS

def make_target_function(formula_name, Y, B):
    # capture the name outside the traced function -> avoid passing strings to JAX
    def F(X_in):
        return jnp.sum(jax_keops_convolution(formula_name, X_in, Y, B))
    return F

def test_derivate_fixed():
    print("\n=========================================================")
    print("   TEST FIXED: Forward / First grad / Second grad")
    print("=========================================================")
    key = jax.random.PRNGKey(0)

    M, N, D = 50, 60, 3   # small sizes for correctness tests
    X = jax.random.normal(key, (M, D), dtype=jnp.float32)
    Y = jax.random.normal(jax.random.PRNGKey(1), (N, D), dtype=jnp.float32)
    B = jax.random.normal(jax.random.PRNGKey(2), (N, 1), dtype=jnp.float32)

    formula_name = "conv_gaussienne"
    print(f"Testing formula: {formula_name} -> '{FORMULA_STRINGS[FORMULAS[formula_name]]}'")

    F = make_target_function(formula_name, Y, B)

    # Forward
    t0 = time.time()
    scalar = F(X)
    jax.block_until_ready(scalar)
    t_forward = (time.time() - t0) * 1000.0
    print(f"Forward scalar sum: {float(scalar):.6f} (time {t_forward:.2f} ms)")

    # Gradient (first order)
    t0 = time.time()
    g = jax.grad(F)(X)
    jax.block_until_ready(g)
    t_grad = (time.time() - t0) * 1000.0
    print(f"Gradient shape: {g.shape}  (time {t_grad:.2f} ms)")

    # Reference gradient with pure-jax (small sizes)
    def naive_scalar(X_):
        dist_sq = jnp.sum((X_[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        K = jnp.exp(-dist_sq)
        return jnp.sum(K @ B)

    g_ref = jax.grad(naive_scalar)(X)
    diff = float(jnp.linalg.norm(g - g_ref))
    print(f"||grad - grad_ref|| = {diff:.6e}")
    np.testing.assert_allclose(np.array(g), np.array(g_ref), rtol=1e-4, atol=1e-5)
    print("Gradient validated.")

    # Hessian (2nd order)
    print("\nAttempting second derivative: grad(grad(F))")
    t0 = time.time()
    try:
        # Ceci va maintenant appeler la règle JVP personnalisée, puis la composition VJP-of-VJP
        h = jax.jit(jax.grad(jax.grad(F)))(X) 
        jax.block_until_ready(h)
        t_hess = (time.time() - t0) * 1000.0
        print(f"Hessian-like output shape: {h.shape} (time {t_hess:.2f} ms)")

        # reference
        h_ref = jax.jit(jax.grad(jax.grad(naive_scalar)))(X)
        jax.block_until_ready(h_ref)
        diff_h = float(jnp.linalg.norm(h - h_ref))
        print(f"||h - h_ref|| = {diff_h:.6e}")
        np.testing.assert_allclose(np.array(h), np.array(h_ref), rtol=2e-4, atol=1e-4)
        print("Second derivative validated.")
    except Exception as e:
        print("Second derivative FAILED. Exception:", repr(e))
        raise

    print("\nALL OK.")

if __name__ == "__main__":
    test_derivate_fixed()