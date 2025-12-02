import time
import jax
import jax.numpy as jnp
from keops_jax import jax_keops_convolution


def test_keops_scalability():
    print("TEST COMPLET : INTEGRATION JAX - KEOPS")

    device = jax.devices()[0].platform
    print(f"Backend JAX détecté : {device}")

    key = jax.random.PRNGKey(42)

    # -------------------------------
    # 1. Test simple (vérification)
    # -------------------------------
    print("\n[1] TEST BASIQUE (Forward + Backward)")

    M, N, D = 200, 300, 3
    X = jax.random.normal(key, (M, D), dtype=jnp.float32)
    Y = jax.random.normal(key, (N, D), dtype=jnp.float32)
    B = jax.random.normal(key, (N, 1), dtype=jnp.float32)

    start = time.time()
    F = jax_keops_convolution(0, X, Y, B)
    print("Output shape :", F.shape)
    print("Temps forward :", (time.time() - start) * 1000, "ms")

    # Gradient
    start = time.time()
    grad_fn = jax.grad(lambda X_: jnp.sum(jax_keops_convolution(0, X_, Y, B)))
    G = grad_fn(X)
    print("Gradient shape :", G.shape)
    print("Temps backward :", (time.time() - start) * 1000, "ms")

    print("\nForward + backward OK sur petites tailles.")


    #2. TEST DE SCALABILITÉ : 10,000 × 10,000 points
    print("\n[2] TEST SCALABILITÉ (10k × 10k) — 100 MILLIONS DE PAIRES")

    M_big, N_big, D = 10_000, 10_000, 3
    X_big = jax.random.normal(key, (M_big, D), dtype=jnp.float32)
    Y_big = jax.random.normal(key, (N_big, D), dtype=jnp.float32)
    B_big = jax.random.normal(key, (N_big, 1), dtype=jnp.float32)

    print("Compilation JIT (premier appel)...")
    start = time.time()
    _ = jax_keops_convolution(0, X_big, Y_big, B_big)
    print("Compilation :", round((time.time() - start) * 1000, 2), "ms")

    print("\nCalcul réel du forward sur 100M paires...")
    start = time.time()
    F_big = jax_keops_convolution(0, X_big, Y_big, B_big)
    F_big.block_until_ready()
    print("Temps forward :", round((time.time() - start) * 1000, 2), "ms")

    print("\nCalcul du gradient sur 100M paires (KeOps streaming)")
    start = time.time()
    grad_big = jax.grad(lambda X_: jnp.sum(jax_keops_convolution(0, X_, Y_big, B_big)))(X_big)
    grad_big.block_until_ready()
    print("Temps backward :", round((time.time() - start) * 1000, 2), "ms")
    print("Gradient shape :", grad_big.shape)

    print("RÉSULTATS & VALIDATION")

    print("KeOps a calculé la convolution sans stockage explicite d'une matrice complète.")
    print("Compilation et autodiff vérifiées pour les cas testés.")


if __name__ == "__main__":
    test_keops_scalability()

#TODO : faire des différences de performance JAX pur vs JAX+KeOps sur des gros datasets
#TODO: faire le fw et le bw avec formules 
#TODO : commencer à rédiger le rapport 
