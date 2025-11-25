from x_square import *
import jax
import jax.numpy as jnp
import numpy as np

x = jnp.linspace(-1, 1, 8, dtype=jnp.float32).reshape(2,4)

print("FFI:", x_square(x))
print("REF:", x * x)

np.testing.assert_allclose(x_square(x), x * x)
print("✓ Basic OK")

# vmap
np.testing.assert_allclose(jax.vmap(x_square)(x), jax.vmap(lambda t: t*t)(x))
print("✓ vmap OK")

# sequential vmap
jax.vmap(x_square_sequential)(x)
print("✓ sequential vmap OK")

# gradients
val, grad = jax.value_and_grad(lambda t: jnp.sum(x_square_vjp(t)))(x)
print("grad:", grad)
np.testing.assert_allclose(grad, 2*x)
print("✓ grad OK")
