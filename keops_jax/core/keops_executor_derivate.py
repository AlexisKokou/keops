import numpy as np
import torch
from pykeops.torch import Genred

KERNEL_CACHE = {}

def to_np(x):
    return x.detach().cpu().numpy()


# KERNELS

def make_kernel(formula, D):
    key = (formula, D, 0)
    if key in KERNEL_CACHE:
        return KERNEL_CACHE[key]

    aliases = [
        f"X = Vi({D})",
        f"Y = Vj({D})",
        "B = Vj(1)"
    ]
    kernel = Genred(formula, aliases, reduction_op="Sum", axis=1)
    KERNEL_CACHE[key] = kernel
    return kernel


def make_grad1_kernel(formula, D):
    key = (formula, D, 1)
    if key in KERNEL_CACHE:
        return KERNEL_CACHE[key]
    formula_grad = f"Grad({formula}, X, V0)"

    aliases = [
        f"X = Vi({D})",
        f"Y = Vj({D})",
        "B = Vj(1)",
        "V0 = Vi(1)"
    ]

    kernel = Genred(formula_grad, aliases, reduction_op="Sum", axis=1)
    KERNEL_CACHE[key] = kernel
    return kernel


def make_grad2_kernel(formula, D):
    key = (formula, D, 2)
    if key in KERNEL_CACHE:
        return KERNEL_CACHE[key]

    formula_grad2 = f"Grad(Grad({formula}, X, V0), X, V1)"

    aliases = [
        f"X = Vi({D})",
        f"Y = Vj({D})",
        "B = Vj(1)",
        "V0 = Vi(1)",
        "V1 = Vi(1)"
    ]

    kernel = Genred(formula_grad2, aliases, reduction_op="Sum", axis=1)
    KERNEL_CACHE[key] = kernel
    return kernel


# 1. FORWARD

def keops_forward(fid, X, Y, B, FORMULA_STRINGS):
    formula = FORMULA_STRINGS[fid]
    D = X.shape[1]

    kernel = make_kernel(formula, D)

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)

    return to_np(kernel(X_t, Y_t, B_t))


# 2. GRAD 1

def keops_grad1(fid, X, Y, B, G, FORMULA_STRINGS):
    formula = FORMULA_STRINGS[fid]
    D = X.shape[1]

    kernel = make_grad1_kernel(formula, D)

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)
    G_t = torch.tensor(G, dtype=torch.float32)

    return to_np(kernel(X_t, Y_t, B_t, G_t))


# 3. GRAD 2


def keops_grad2(fid, X, Y, B, G1, G2, FORMULA_STRINGS):
    formula = FORMULA_STRINGS[fid]
    D = X.shape[1]

    kernel = make_grad2_kernel(formula, D)

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    B_t = torch.tensor(B, dtype=torch.float32)
    G1_t = torch.tensor(G1, dtype=torch.float32)
    G2_t = torch.tensor(G2, dtype=torch.float32)

    return to_np(kernel(X_t, Y_t, B_t, G1_t, G2_t))
