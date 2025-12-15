from .keops_executor import keops_forward, make_kernel
from .device_utils import get_jax_backend, safe_jax_to_numpy, numpy_to_jax, check_gpu_available
from .formulas import FORMULAS, FORMULA_STRINGS
from .keops_autodiff import DERIVATIVE_GENERATOR, KeOpsDerivativeGenerator
from .keops_functions import (
    conv_gaussienne, conv_cauchy, mat_vec_mult, copy_B,
)
# Importez depuis le nouveau fichier
from .keops_higher_order_final import (
    higher_order_gaussian,
    higher_order_cauchy,
    higher_order_mat_vec_mult,
    higher_order_copy_B,
    HigherOrderKeOpsFunction,
)

__all__ = [
    'keops_forward',
    'make_kernel',
    'get_jax_backend',
    'safe_jax_to_numpy',
    'numpy_to_jax',
    'check_gpu_available',
    'FORMULAS',
    'FORMULA_STRINGS',
    'DERIVATIVE_GENERATOR',
    'KeOpsDerivativeGenerator',
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