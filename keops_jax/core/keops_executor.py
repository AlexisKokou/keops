import numpy as np
import torch
from pykeops.torch import LazyTensor

# Convert a torch tensor to a NumPy array on CPU (detached from computation graph)
def to_np(x):
    return x.detach().cpu().numpy()


def keops_forward(formula_id, X_np, Y_np, B_np):
    """
    Calcul forward utilisant PyKeOps via LazyTensor.
    Entrées (NumPy):
      - X_np: (N, D) points i
      - Y_np: (M, D) points j
      - B_np: (M, C) coefficients associés aux j (peut être vecteur ou matrice)
    Retour:
      - out: résultat NumPy de shape (N, C) correspondant à sum_j expr(i,j)
    """

    # Choix du device (préférence GPU si disponible)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convertir en tenseurs torch sur le device choisi (float32 pour KeOps)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    Y = torch.tensor(Y_np, dtype=torch.float32, device=device)
    B = torch.tensor(B_np, dtype=torch.float32, device=device)

    # LazyTensor attend des dimensions de type "i" et "j":
    # X[:, None, :] -> shape (N,1,D) devient indexé par i
    # Y[None, :, :] -> shape (1,M,D) devient indexé par j
    x_i = LazyTensor(X[:, None, :])
    y_j = LazyTensor(Y[None, :, :])
    # b_j reste un tenseur classique mais aligné en dimension j via [None, :, :]
    b_j = B[None, :, :]

    # ---- SELECT FORMULA ----
    # Ici on implémente une exponentielle gaussienne pondérée: exp(-||x-y||^2) * b_j
    if formula_id == 0:
        # sqnorm2() calcule la norme carrée euclidienne le long de la dernière dimension
        expr = (-(x_i - y_j).sqnorm2()).exp() * b_j
    else:
        raise ValueError(f"Unknown formula ID: {formula_id}")

    # Somme sur l'indice j -> résultat de shape (N, C)
    out = expr.sum(dim=1)
    return to_np(out)


def keops_backward(formula_id, X_np, Y_np, B_np, G_np):
    """
    Calcul du gradient de la sortie F par rapport à X (dF/dX) via backprop.
    Entrées (NumPy):
      - X_np: (N, D) points i (on veut le gradient par rapport à X)
      - Y_np: (M, D)
      - B_np: (M, C)
      - G_np: (N, C) gradient amont (grad_outputs) pour la somme sur j
    Retour:
      - dX: NumPy array (N, D) = dF/dX * G (contraction via grad_outputs)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # X doit require_grad pour pouvoir calculer son gradient
    X = torch.tensor(X_np, dtype=torch.float32, requires_grad=True, device=device)
    Y = torch.tensor(Y_np, dtype=torch.float32, device=device)
    B = torch.tensor(B_np, dtype=torch.float32, device=device)
    G = torch.tensor(G_np, dtype=torch.float32, device=device)

    x_i = LazyTensor(X[:, None, :])
    y_j = LazyTensor(Y[None, :, :])
    b_j = B[None, :, :]

    if formula_id == 0:
        expr = (-(x_i - y_j).sqnorm2()).exp() * b_j
    else:
        raise ValueError(f"formule id inconnue: {formula_id}")

    # F est la somme sur j: forme (N, C)
    F = expr.sum(dim=1)

    # Calcul du gradient de F par rapport à X.
    # grad_outputs correspond au gradient amont G (mêmes dimensions que F)
    (dX,) = torch.autograd.grad(F, X, grad_outputs=G)
    return to_np(dX)
