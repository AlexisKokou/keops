# jax_interface_recursive.py

import jax
import jax.numpy as jnp
from .keops_callback import keops_callback
from .formulas import FORMULAS, FORMULA_STRINGS

_JAX_FUNC_CACHE = {}

def make_keops_function(formula_name):
    formula_id = FORMULAS[formula_name]

    @jax.custom_jvp
    def keops_fun(X, Y, B):
        return keops_callback(
            formula_id,
            X, Y, B,
            direction_vectors=(),
            FORMULA_STRINGS=FORMULA_STRINGS
        )

    @keops_fun.defjvp
    def keops_fun_jvp(primals, tangents):
        X, Y, B = primals
        Xdot, Ydot, Bdot = tangents

        # ---- primal ----
        primal = keops_fun(X, Y, B)

        # ---- collect directions (ONLY X) ----
        directions = []
        if Xdot is not None:
            directions.append(Xdot)

        # ---- tangent ----
        if len(directions) == 0:
            tangent = jnp.zeros_like(primal)
        else:
            tangent = keops_callback(
                formula_id,
                X, Y, B,
                direction_vectors=tuple(directions),
                FORMULA_STRINGS=FORMULA_STRINGS
            )

        return primal, tangent

    return keops_fun


def jax_keops_convolution(formula_name, X, Y, B):
    if formula_name not in _JAX_FUNC_CACHE:
        _JAX_FUNC_CACHE[formula_name] = make_keops_function(formula_name)
    return _JAX_FUNC_CACHE[formula_name](X, Y, B)
