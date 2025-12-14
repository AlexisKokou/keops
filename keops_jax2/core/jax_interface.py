# keops_jax/core/jax_interface.py
"""
Interface JAX pour KeOps - Version compatible avec dérivées d'ordre supérieur.
"""
import jax.numpy as jnp
from .advanced_interface import jax_keops_convolution


# API compatible avec l'ancien code
__all__ = ['jax_keops_convolution']

# Note: La fonction jax_keops_convolution est importée depuis advanced_interface
# Elle a la même signature que l'ancienne version mais supporte maintenant
# les dérivées d'ordre supérieur par défaut.