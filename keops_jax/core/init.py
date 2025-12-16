# keops_jax/core/__init__.py
"""Package principal KeOps-JAX"""

from .formulas import FORMULAS, FORMULA_STRINGS
from .keops_executor_nth_order import keops_nth_order, make_nth_order_kernel
from .jax_interface_nth_order import (
    jax_keops_convolution,
    jax_keops_gradient,
    jax_keops_hessian,
    jax_keops_third_order,
    jax_keops_convolution_scalar,
    make_vector_keops_function,
    make_scalar_keops_function
)
from .device_utils import jax_to_numpy, numpy_to_jax, check_gpu_available

__version__ = "1.0.0"
__all__ = [
    'FORMULAS',
    'FORMULA_STRINGS',
    'keops_nth_order',
    'make_nth_order_kernel',
    'jax_keops_convolution',
    'jax_keops_gradient',
    'jax_keops_hessian',
    'jax_keops_third_order',
    'jax_keops_convolution_scalar',
    'make_vector_keops_function',
    'make_scalar_keops_function',
    'jax_to_numpy',
    'numpy_to_jax',
    'check_gpu_available'
]