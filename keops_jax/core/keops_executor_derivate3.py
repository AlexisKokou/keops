# keops_executor.py

import numpy as np
from pykeops.numpy import Genred

_KERNEL_CACHE = {}

def make_nth_order_kernel(formula, D, order):
    key = (formula, D, order)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    aliases = [
        f"X = Vi({D})",
        f"Y = Vj({D})",
        "B = Vj(1)",
    ]

    expr = formula
    for k in range(order):
        expr = f"Grad({expr}, X, V{k})"
        aliases.append(f"V{k} = Vi({D})")

    kernel = Genred(expr, aliases, reduction_op="Sum", axis=1)
    _KERNEL_CACHE[key] = kernel
    return kernel


def keops_nth_order(formula_id, X, Y, B, *directions, FORMULA_STRINGS):
    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)
    B = np.asarray(B, np.float32)
    directions = [np.asarray(v, np.float32) for v in directions]

    D = X.shape[1]
    order = len(directions)

    formula = FORMULA_STRINGS[formula_id]
    kernel = make_nth_order_kernel(formula, D, order)

    out = kernel(X, Y, B, *directions)

    # projection scalaire pour ordre >= 2 (directionnel)
    if order >= 2 and out.ndim == 2 and out.shape[1] > 1:
        out = out.sum(axis=1, keepdims=True)

    return out
