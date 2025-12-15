"""
KeOps-JAX: Interface JAX pour le backend KeOps avec support complet des dérivées d'ordre supérieur.
"""

from .core import (
    conv_gaussienne,
    conv_cauchy,
    mat_vec_mult,
    copy_B,
    higher_order_gaussian,
    higher_order_cauchy,
    higher_order_mat_vec_mult,
    higher_order_copy_B,
    HigherOrderKeOpsFunction,
)

__all__ = [
    'conv_gaussienne',
    'conv_cauchy',
    'mat_vec_mult',
    'copy_B',
    'higher_order_gaussian',
    'higher_order_cauchy',
    'higher_order_mat_vec_mult',
    'higher_order_copy_B',
    'HigherOrderKeOpsFunction',
]