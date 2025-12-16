import jax
import jax.numpy as jnp
from keops_jax.jax_interface_recursive import jax_keops_convolution

X = jnp.ones((10, 2))
Y = jnp.ones((20, 2))
B = jnp.ones((20, 1))

def f(X):
    return jax_keops_convolution("conv_gaussienne", X, Y, B).sum()

# 1er ordre
g = jax.grad(f)(X)

# 2e ordre (Hessien directionnel)
v = jnp.ones_like(X)
Hv = jax.jvp(jax.grad(f), (X,), (v,))[1]

# 3e ordre
_, d3 = jax.jvp(
    lambda x: jax.jvp(jax.grad(f), (x,), (v,))[1],
    (X,),
    (v,)
)
print(f"Shape 3e ordre: {d3.shape}")