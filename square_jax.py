import ctypes
import jax
import jax.numpy as jnp
import numpy as np
from jax.ffi import ffi_call, register_ffi_target, pycapsule
from jax import custom_vjp, ShapeDtypeStruct

lib = ctypes.CDLL("./square.dylib")
jax.ffi.register_ffi_target(
    "square", jax.ffi.pycapsule(lib.square), platform="cpu")

# Primitive FFI JAX (forward)
def square_ffi(x):
    # La spécification de sortie est un scalaire de même dtype que l'entrée x
    out_spec = ShapeDtypeStruct((), x.dtype)
    return ffi_call(
        "square_f32",         # Le nom de la cible FFI enregistrée ci-dessus
        out_spec,
        vmap_method="broadcast_all"
    )(x)

# Ajouter le backward via custom_vjp
@custom_vjp
def square(x):
    return square_ffi(x)

def square_fwd(x):
    y = square_ffi(x)
    return y, x  # y est le résultat, x est le résidu (nécessaire pour le backward)

def square_bwd(res, g):
    (x,) = (res,)  # Récupère le résidu x
    # Dérivée de x^2 est 2x. Multipliée par le gradient amont g.
    return (2.0 * x * g,)

square.defvjp(square_fwd, square_bwd)

# Test
if __name__ == "__main__":
    # Correction: Utiliser un array JAX, pas un float Python standard
    x = jnp.array(3.0, dtype=jnp.float32)

    # Test de la fonction
    print(f"Entrée x = {x}")
    print("square(3) =", square(x))

    # Test de la dérivée
    print("grad square(3) =", jax.grad(square)(x))
    # Résultat attendu : 2 * 3.0 = 6.0