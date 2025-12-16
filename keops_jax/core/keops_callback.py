# keops_callback.py

import numpy as np
import jax.numpy as jnp
from .keops_executor_derivate3 import keops_nth_order

def keops_callback(
    formula_id,
    X, Y, B,
    direction_vectors=(),
    FORMULA_STRINGS=None
):
    X_np = np.asarray(X)
    Y_np = np.asarray(Y)
    B_np = np.asarray(B)
    dirs_np = [np.asarray(v) for v in direction_vectors]

    out = keops_nth_order(
        formula_id,
        X_np, Y_np, B_np,
        *dirs_np,
        FORMULA_STRINGS=FORMULA_STRINGS
    )

    return jnp.asarray(out)
