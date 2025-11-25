import jax
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import ctypes

# Load lib
lib_path = next(Path("ffi").glob("libx_square*"))
x_square_lib = ctypes.cdll.LoadLibrary(lib_path)

# Register handler
jax.ffi.register_ffi_target(
    "x_square",
    jax.ffi.pycapsule(x_square_lib.XSquare),
    platform="cpu",
)

def x_square(x):
    if x.dtype != jnp.float32:
        raise ValueError("Only float32 supported")

    call = jax.ffi.ffi_call(
        "x_square",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )
    return call(x)


# Sequential version
def x_square_sequential(x):
    return jax.ffi.ffi_call(
        "x_square",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="sequential",
    )(x)


# Custom VJP (auto-diff) 
def x_square_fwd(x):
    y = x_square(x)
    return y, x

def x_square_bwd(x, g):
    return (2 * x * g,)

x_square_vjp = jax.custom_vjp(x_square)
x_square_vjp.defvjp(x_square_fwd, x_square_bwd)


#Cross-platform CPU/GPU stub
def x_square_cross(x):
    def impl(target):
        return lambda x: jax.ffi.ffi_call(
            target,
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )(x)

    return jax.lax.platform_dependent(
        x,
        cpu=impl("x_square"),
        cuda=impl("x_square_cuda"),  # stub
    )
