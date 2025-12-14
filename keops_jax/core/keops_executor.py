# keops_executor.py

import numpy as np
from pykeops.numpy import Genred


# GENRED factory
def make_kernel(formula_string, D):
    aliases = [
        f"X = Vi({D})",
        f"Y = Vj({D})",
        "B = Vj(1)",
    ]
    return Genred(formula_string, aliases, reduction_op="Sum", axis=1)


def make_grad_kernel(formula_string, D):
    grad_formula = f"""
        Grad( {formula_string},
              X, 
              Vx )
    """
    aliases = [
        f"X = Vi({D})",
        f"Y = Vj({D})",
        "B = Vj(1)",
        f"Vx = Vi(1)"  # gradient entrant G
    ]
    return Genred(grad_formula, aliases, reduction_op="Sum", axis=1)



# FORWARD
def keops_forward(formula_id, X_np, Y_np, B_np, FORMULA_STRINGS):
    # Convertir les tableaux JAX en NumPy avant de passer à PyKeOps
    X_np = np.asarray(X_np)
    Y_np = np.asarray(Y_np)
    B_np = np.asarray(B_np)
    
    N, D = X_np.shape

    formula = FORMULA_STRINGS[formula_id]
    kernel = make_kernel(formula, D)

    out = kernel(X_np, Y_np, B_np)
    return out


# BACKWARD
def keops_backward(formula_id, X_np, Y_np, B_np, G_np, FORMULA_STRINGS):
    # Convertir les tableaux JAX en NumPy avant de passer à PyKeOps
    X_np = np.asarray(X_np)
    Y_np = np.asarray(Y_np)
    B_np = np.asarray(B_np)
    G_np = np.asarray(G_np)
    
    N, D = X_np.shape

    formula = FORMULA_STRINGS[formula_id]
    grad_kernel = make_grad_kernel(formula, D)

    # Execute le gradient symbolique de keOps
    dX = grad_kernel(X_np, Y_np, B_np, G_np)

    return dX