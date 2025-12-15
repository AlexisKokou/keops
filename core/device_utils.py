# keops_jax/core/device_utils.py
"""
Utilitaires pour JAX.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

def get_jax_backend() -> str:
    """Retourne le backend JAX actuel."""
    return jax.default_backend()

def safe_jax_to_numpy(jax_array: Any) -> np.ndarray:
    """Convertit un array JAX en array NumPy de manière sûre."""
    if isinstance(jax_array, np.ndarray):
        return jax_array
    return np.asarray(jax_array)

def numpy_to_jax(np_array: np.ndarray) -> jnp.ndarray:
    """Convertit un array NumPy en array JAX."""
    return jnp.asarray(np_array)

def check_gpu_available() -> bool:
    """Vérifie si JAX peut utiliser le GPU."""
    try:
        return any(device.platform == 'gpu' for device in jax.devices())
    except:
        return False