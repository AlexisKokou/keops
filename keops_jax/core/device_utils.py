# keops_jax/core/device_utils.py
"""Utilitaires pour la gestion des devices"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

def jax_to_numpy(jax_array: jnp.ndarray) -> np.ndarray:
    """Convertit un array JAX en array NumPy"""
    return np.asarray(jax_array)

def numpy_to_jax(np_array: np.ndarray) -> jnp.ndarray:
    """Convertit un array NumPy en array JAX"""
    return jnp.array(np_array)

def check_gpu_available() -> bool:
    """Vérifie si JAX peut utiliser le GPU"""
    try:
        return jax.default_backend() == 'gpu'
    except:
        return False

def get_backend_info() -> str:
    """Retourne des informations sur le backend"""
    backend = jax.default_backend()
    device_count = jax.device_count()
    return f"Backend: {backend}, Devices: {device_count}"

def synchronize_device():
    """Synchronise le device (utile pour les mesures de temps)"""
    if check_gpu_available():
        # Force synchronization on GPU
        jax.devices()[0].synchronize_all_activity()