
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional

def get_jax_backend() -> str:
    """Retourne le backend JAX actuel."""
    return jax.default_backend()

def jax_to_numpy(jax_array) -> np.ndarray:  
    """
    Convertit un array JAX en array NumPy.
    (C'est gratuit et sans copie si possible)
    """
    return np.asarray(jax_array)

def numpy_to_jax(np_array) -> jnp.ndarray:
    """
    Convertit un array NumPy en array JAX.
    """
    return jnp.array(np_array)

def check_gpu_available() -> bool:
    """Vérifie si JAX peut utiliser le GPU."""
    backend = get_jax_backend()
    return backend == 'gpu'

def synchronize_if_needed():
    """Synchronise les opérations si nécessaire."""
    if check_gpu_available():
        # Avec JAX, block_until_ready() assure la synchronisation
        pass