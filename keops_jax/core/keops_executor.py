import numpy as np
import torch
from pykeops.torch import Genred


def to_np(x):
    return x.detach().cpu().numpy()


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_np, dtype=torch.float32, device=device)
    B = torch.tensor(B_np, dtype=torch.float32, device=device)

    N, D = X.shape

    formula = FORMULA_STRINGS[formula_id]
    kernel = make_kernel(formula, D)

    out = kernel(X, Y, B)
    return to_np(out)


# BACKWARD
def keops_backward(formula_id, X_np, Y_np, B_np, G_np, FORMULA_STRINGS):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_np, dtype=torch.float32, device=device)
    B = torch.tensor(B_np, dtype=torch.float32, device=device)
    G = torch.tensor(G_np, dtype=torch.float32, device=device)

    N, D = X.shape

    formula = FORMULA_STRINGS[formula_id]
    grad_kernel = make_grad_kernel(formula, D)

    # Execute le gradient symbolique de keOps
    dX = grad_kernel(X, Y, B, G)

    return to_np(dX)