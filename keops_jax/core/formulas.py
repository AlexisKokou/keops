# Dictionnaire lisible par l'utilisateur JAX

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
    
    # 2. (X | Y) * B : Produit scalaire de X_i et Y_j
    2: "(X | Y) * B", # Multiplie le produit scalaire par le vecteur B_j avant la sommation.
    
    # 3. B : Copie simplement la valeur du vecteur Vj(1) sur Vi
    3: "B",
}