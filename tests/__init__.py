# keops_jax/__init__.py
"""
API principale pour les utilisateurs JAX.
"""
from .core.keops_jax import (
    conv_gaussienne,
    conv_cauchy,
    mat_vec_mult,
    copy_B,
    gaussian_kernel,
    cauchy_kernel,
    dot_product_kernel,
    copy_kernel,
    KeOpsJaxKernel,
    keops_function,
)

__all__ = [
    'conv_gaussienne',
    'conv_cauchy', 
    'mat_vec_mult',
    'copy_B',
    'gaussian_kernel',
    'cauchy_kernel',
    'dot_product_kernel',
    'copy_kernel',
    'KeOpsJaxKernel',
    'keops_function',
]