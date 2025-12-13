# keops_jax/core/advanced_interface.py
import jax
import jax.numpy as jnp
import functools
from typing import Optional, Callable, Tuple
from .device_utils import jax_to_numpy
from .formulas import FORMULAS, FORMULA_STRINGS
from .keops_executor import keops_forward  # CHANGÉ : seulement keops_forward

# ... le reste du code reste inchangé ...
class KeOpsFunctionRegistry:
    """Registre intelligent qui maintient les implémentations optimisées."""
    
    def __init__(self):
        self._forward_cache = {}
        self._vjp_cache = {}
        self._jvp_cache = {}
    
    def get_cached_forward(self, formula_id: int, D: int):
        """Retourne une fonction forward compilée avec mémoization."""
        key = (formula_id, D)
        if key not in self._forward_cache:
            def forward_impl(X, Y, B):
                M = X.shape[0]
                out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)
                
                def fw_callback(X_, Y_, B_):
                    # Conversion JAX → NumPy
                    X_np = jax_to_numpy(X_)  # CHANGÉ ICI
                    Y_np = jax_to_numpy(Y_)  # CHANGÉ ICI
                    B_np = jax_to_numpy(B_)  # CHANGÉ ICI
                    return keops_forward(formula_id, X_np, Y_np, B_np, FORMULA_STRINGS)
                
                return jax.pure_callback(fw_callback, out_shape, X, Y, B)
            
            self._forward_cache[key] = forward_impl
        return self._forward_cache[key]
    
    # ... le reste du code reste identique ...
    
    # ... (le reste du code reste identique, les fonctions VJP/JVP analytiques ne changent pas)
    
    def get_cached_jvp(self, formula_id: int, D: int):
        """Retourne une fonction JVP analytique en JAX."""
        key = (formula_id, D)
        if key not in self._jvp_cache:
            if formula_id == 0:  # Gaussian kernel
                def jvp_gaussian(X, Y, B, dX, dY, dB):
                    diff = X[:, None, :] - Y[None, :, :]
                    d_diff = dX[:, None, :] - dY[None, :, :]
                    
                    sq_dist = jnp.sum(diff**2, axis=-1)
                    d_sq_dist = 2 * jnp.sum(diff * d_diff, axis=-1)
                    
                    K = jnp.exp(-sq_dist)
                    dK = -K * d_sq_dist
                    
                    term1 = (dK @ B)
                    term2 = (K @ dB)
                    return term1 + term2
                
                self._jvp_cache[key] = jvp_gaussian
                
            elif formula_id == 1:  # Cauchy kernel
                def jvp_cauchy(X, Y, B, dX, dY, dB):
                    diff = X[:, None, :] - Y[None, :, :]
                    d_diff = dX[:, None, :] - dY[None, :, :]
                    
                    sq_dist = jnp.sum(diff**2, axis=-1)
                    d_sq_dist = 2 * jnp.sum(diff * d_diff, axis=-1)
                    
                    K = 1.0 / (1.0 + sq_dist)
                    dK = -K**2 * d_sq_dist
                    
                    term1 = (dK @ B)
                    term2 = (K @ dB)
                    return term1 + term2
                
                self._jvp_cache[key] = jvp_cauchy
                
            elif formula_id == 2:  # Matrix-vector product
                def jvp_matvec(X, Y, B, dX, dY, dB):
                    term1 = (dX @ Y.T) @ B
                    term2 = (X @ dY.T) @ B
                    term3 = (X @ Y.T) @ dB
                    return term1 + term2 + term3
                
                self._jvp_cache[key] = jvp_matvec
                
            elif formula_id == 3:  # Copy B
                def jvp_copy(X, Y, B, dX, dY, dB):
                    return jnp.sum(dB) * jnp.ones((X.shape[0], 1), dtype=X.dtype)
                
                self._jvp_cache[key] = jvp_copy
                
            else:
                raise ValueError(f"Pas d'implémentation JVP pour formula_id {formula_id}")
        
        return self._jvp_cache[key]

# Registre global
REGISTRY = KeOpsFunctionRegistry()


def create_vector_keops_function(formula_id: int):
    """
    Crée une fonction vectorielle KeOps (retourne (M, 1)).
    Cette fonction a à la fois JVP et VJP définis.
    """
    
    @jax.custom_jvp
    @jax.custom_vjp
    def keops_func_vector(X, Y, B):
        """Forward via KeOps callback."""
        D = X.shape[1]
        forward_fn = REGISTRY.get_cached_forward(formula_id, D)
        return forward_fn(X, Y, B)
    
    # Définition des règles JVP
    @keops_func_vector.defjvp
    def keops_func_vector_jvp(primals, tangents):
        """JVP rule: utilise l'implémentation analytique en JAX."""
        X, Y, B = primals
        dX, dY, dB = tangents
        
        # Calcul forward
        output = keops_func_vector(X, Y, B)
        
        # Calcul JVP analytique
        D = X.shape[1]
        jvp_fn = REGISTRY.get_cached_jvp(formula_id, D)
        d_output = jvp_fn(X, Y, B, dX, dY, dB)
        
        return output, d_output
    
    # Définition des règles VJP
    def vector_fwd(X, Y, B):
        """Forward pass pour custom_vjp."""
        return keops_func_vector(X, Y, B), (X, Y, B)
    
    def vector_bwd(res, G):
        """Backward pass pour custom_vjp."""
        X, Y, B = res
        D = X.shape[1]
        vjp_fn = REGISTRY.get_cached_vjp(formula_id, D)
        dX, dY, dB = vjp_fn(X, Y, B, G)
        return (dX, dY, dB)
    
    keops_func_vector.defvjp(vector_fwd, vector_bwd)
    
    return keops_func_vector


# Fonctions vectorielles internes
_vector_gaussian = create_vector_keops_function(0)
_vector_cauchy = create_vector_keops_function(1)
_vector_matvec = create_vector_keops_function(2)
_vector_copy = create_vector_keops_function(3)


# ============================================================================
# FONCTIONS POUR L'UTILISATEUR - SIMPLES ET DIRECTES
# ============================================================================

def make_keops_function(vector_func, formula_id: int):
    """
    Transforme une fonction vectorielle en fonction scalaire
    que l'utilisateur peut utiliser avec jax.grad directement.
    """
    
    @jax.custom_jvp
    @jax.custom_vjp
    def keops_func(X, Y, B):
        """
        Fonction KeOps SCALAIRE.
        L'utilisateur peut faire jax.grad sur cette fonction directement.
        """
        # Retourne la somme du résultat vectoriel → scalaire
        return jnp.sum(vector_func(X, Y, B))
    
    # Règle JVP pour la fonction scalaire
    @keops_func.defjvp
    def keops_func_jvp(primals, tangents):
        X, Y, B = primals
        dX, dY, dB = tangents
        
        # Calcul forward (scalaire)
        output = keops_func(X, Y, B)
        
        # Calcul du JVP (scalaire)
        D = X.shape[1]
        jvp_fn = REGISTRY.get_cached_jvp(formula_id, D)
        
        # JVP vectoriel
        vector_output = vector_func(X, Y, B)
        d_vector_output = jvp_fn(X, Y, B, dX, dY, dB)
        
        # Somme pour obtenir le JVP scalaire
        d_output = jnp.sum(d_vector_output)
        
        return output, d_output
    
    # Règle VJP pour la fonction scalaire
    def scalar_fwd(X, Y, B):
        """Forward pass pour la fonction scalaire (pour VJP)."""
        return keops_func(X, Y, B), (X, Y, B)
    
    def scalar_bwd(res, g):
        """Backward pass pour la fonction scalaire (pour VJP)."""
        X, Y, B = res
        # g est un scalaire (le gradient de la sortie scalaire)
        
        # On appelle le backward de la fonction vectorielle
        # avec G = g (un scalaire étendu à la forme de sortie)
        vector_output = vector_func(X, Y, B)
        G = g * jnp.ones_like(vector_output)  # (M, 1)
        
        # Calcul VJP vectoriel
        D = X.shape[1]
        vjp_fn = REGISTRY.get_cached_vjp(formula_id, D)
        dX, dY, dB = vjp_fn(X, Y, B, G)
        
        return (dX, dY, dB)
    
    keops_func.defvjp(scalar_fwd, scalar_bwd)
    
    return keops_func


# FONCTIONS QUE L'UTILISATEUR IMPORTE
keops_gaussian = make_keops_function(_vector_gaussian, 0)
keops_cauchy = make_keops_function(_vector_cauchy, 1)
keops_matvec = make_keops_function(_vector_matvec, 2)
keops_copy = make_keops_function(_vector_copy, 3)

# Alias pour compatibilité
jax_keops_gaussian = keops_gaussian
jax_keops_cauchy = keops_cauchy
jax_keops_matvec = keops_matvec
jax_keops_copy = keops_copy