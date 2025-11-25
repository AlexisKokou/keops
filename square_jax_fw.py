import ctypes
import jax
import jax.numpy as jnp
import numpy as np
from jax.ffi import ffi_call, register_ffi_target, pycapsule
from jax import custom_vjp, ShapeDtypeStruct

lib = ctypes.CDLL("./square.dylib")
jax.ffi.register_ffi_target(
    "square", jax.ffi.pycapsule(lib.square), platform="cpu")

def square(x):
    # Only the `float32` version is implemented by the FFI target.
    if x.dtype != jnp.float32:
        raise ValueError("Only the float32 dtype is implemented by square")

    call = jax.ffi.ffi_call(
        # Target name must match the registered custom call target
        "square",
        # Output has same shape and dtype as input
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        # Behavior under vmap
        vmap_method="broadcast_all",
    )

    # No extra static attributes required for this simple example
    return call(x)


# Test that this matches the reference implementation
x = jnp.linspace(-0.5, 0.5, 32, dtype=jnp.float32).reshape((8, 4))
print("Input x:\n", x)
np.testing.assert_allclose(square(x), jnp.square(x), rtol=1e-5)
