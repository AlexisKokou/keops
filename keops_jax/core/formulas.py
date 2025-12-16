# keops_jax/core/formulas.py
"""Définitions des formules KeOps"""

FORMULAS = {
    # 0. Convolution Gaussienne (Originale)
    "conv_gaussienne": 0,
    
    # 1. Convolution de Cauchy (Inverse MultiQuadric)
    "conv_cauchy": 1,
    
    # 2. Multiplication Matrice-Vecteur (LinComb)
    "mat_vec_mult": 2,
    
    # 3. Égalité simple (pour tester la projection de B)
    "copy_B": 3,
}

# Dictionnaire interne KeOps : formula_id -> formula string
FORMULA_STRINGS = {
    # 0. Exp(-SqDist(X,Y)) * B : Noyau Gaussien
    0: "Exp(-SqDist(X,Y)) * B",
    
    # 1. (1 / (1 + SqDist(X,Y))) * B : Noyau de Cauchy / MultiQuadric Inverse - CORRIGÉ
    1: "(1 / (1 + SqDist(X,Y))) * B",
    2: "(X | Y) * B",
    3: "B",
}