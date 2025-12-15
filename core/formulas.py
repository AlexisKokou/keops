# keops_jax/core/formulas.py

FORMULAS = {
    "conv_gaussienne": 0,
    "conv_cauchy": 1,
    "mat_vec_mult": 2,
    "copy_B": 3,
}

FORMULA_STRINGS = {
    0: "Exp(-SqDist(X,Y)) * B",
    1: "(1 / (1 + SqDist(X,Y))) * B",
    2: "(X | Y) * B",
    3: "B",
}