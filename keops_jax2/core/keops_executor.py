
import numpy as np
from pykeops.numpy import Genred


def make_kernel(formula_string, D):
    """
    Crée un noyau KeOps pour le calcul forward.
    
    Args:
        formula_string: Formule KeOps (ex: "Exp(-SqDist(X,Y)) * B")
        D: Dimension des points
    
    Returns:
        Fonction KeOps compilée
    """
    aliases = [
        f"X = Vi({D})",
        f"Y = Vj({D})",
        "B = Vj(1)",
    ]
    return Genred(formula_string, aliases, reduction_op="Sum", axis=1)


def keops_forward(formula_id, X_np, Y_np, B_np, FORMULA_STRINGS):
    """
    Forward pass avec KeOps en NumPy pur.
    
    Args:
        formula_id: ID de la formule (0=gaussian, 1=cauchy, etc.)
        X_np, Y_np, B_np: Arrays NumPy
        FORMULA_STRINGS: Dictionnaire des formules
    
    Returns:
        Résultat du noyau (array NumPy de shape (M, 1))
    """
    N, D = X_np.shape
    
    # Récupère la formule
    formula = FORMULA_STRINGS[formula_id]
    
    # Compile et exécute le noyau
    kernel = make_kernel(formula, D)
    out = kernel(X_np, Y_np, B_np)
    
    return out