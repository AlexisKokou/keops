import jax
import jax.numpy as jnp
import time
from keops_jax.core.jax_interface import jax_keops_convolution

# Dictionnaire des formules et fonctions de référence JAX correspondantes
# La fonction de référence implémente l'opération complète: Sum_j [ K(X_i, Y_j) * B_j ]
TEST_FORMULAS = {
    
    # 0. Convolution Gaussienne: K_ij = Exp(-SqDist(X,Y))
    "conv_gaussienne": {
        "id": 0,
        "ref_fn": lambda X, Y, B: (jnp.exp(-jnp.sum((X[:, None, :] - Y[None, :, :])**2, axis=-1))) @ B
    },
    
    # 1. Convolution de Cauchy: K_ij = 1 / (1 + SqDist(X,Y))
    "conv_cauchy": {
        "id": 1,
        "ref_fn": lambda X, Y, B: (1.0 / (1.0 + jnp.sum((X[:, None, :] - Y[None, :, :])**2, axis=-1))) @ B
    },
    
    # 2. Noyau Linéaire / Matrice-Vecteur: K_ij = X_i . Y_j
    # Note: La formule KeOps est "(X | Y)*B". 
    # La réduction Sum(K_ij) * B_j est équivalente à (X @ Y.T) @ B
    "mat_vec_mult": {
        "id": 2,
        # K = X @ Y.T est la matrice des produits scalaires K_ij = X_i . Y_j
        "ref_fn": lambda X, Y, B: (X @ Y.T) @ B
    },
    
    # 3. Copie B: K_ij = 1. La formule KeOps est "B". La réduction est Sum_j [ 1 * B_j ]
    "copy_B": {
        "id": 3,
        # La réduction Sum_j [B_j] est appliquée à chaque Vi. Le résultat est Sum(B_j) * ones(M, 1)
        "ref_fn": lambda X, Y, B: jnp.sum(B) * jnp.ones((X.shape[0], 1), dtype=X.dtype)
    }
}


def run_test_for_formula(formula_name, X, Y, B, ref_fn):
    """ Exécute les tests Forward et Backward pour une formule donnée. """

    print(f"\n--- FORMULE : {formula_name} ---")
    
    # 1.FORWARD 
    
    # JAX pur
    t0_F_jax = time.time()
    F_jax = ref_fn(X, Y, B)
    F_jax.block_until_ready()
    t_F_jax = (time.time() - t0_F_jax) * 1000
    
    # KEOPS
    t0_F_keops = time.time()
    F_keops = jax_keops_convolution(formula_name, X, Y, B)
    F_keops.block_until_ready()
    t_F_keops = (time.time() - t0_F_keops) * 1000
    
    diff_F = float(jnp.linalg.norm(F_jax - F_keops))
    
    print(f"Forward (JAX: {t_F_jax:.2f} ms | KEOPS: {t_F_keops:.2f} ms)")
    print(f"Forward Diff (‖F_jax - F_keops‖): {diff_F:.6e}")
    assert diff_F < 1e-4, f"Échec du Forward pour {formula_name}: Différence trop grande."


    # 2.BACKWARD 
    
    # Fonction de perte JAX pure pour le gradient
    def loss_fn_naive(X_):
        return jnp.sum(ref_fn(X_, Y, B))

    # Fonction de perte KEOPS pour le gradient
    def loss_fn_keops(X_):
        return jnp.sum(jax_keops_convolution(formula_name, X_, Y, B))

    # Calcul du gradient JAX pur
    t0_B_jax = time.time()
    grad_jax_naive = jax.grad(loss_fn_naive)(X)
    grad_jax_naive.block_until_ready()
    t_B_jax = (time.time() - t0_B_jax) * 1000
    
    # Calcul du gradient KEOPS
    t0_B_keops = time.time()
    grad_keops = jax.grad(loss_fn_keops)(X)
    grad_keops.block_until_ready()
    t_B_keops = (time.time() - t0_B_keops) * 1000

    diff_B = float(jnp.linalg.norm(grad_jax_naive - grad_keops))
    
    print(f"Backward (JAX: {t_B_jax:.2f} ms | KEOPS: {t_B_keops:.2f} ms)")
    print(f"Backward Diff (‖grad_jax - grad_keops‖): {diff_B:.6e}")
    assert diff_B < 1e-4, f"Échec du Backward pour {formula_name}: Différence trop grande."

    print("→ FORWARD et BACKWARD VALIDÉS (Tolérance: 1e-4)")


def test_all_formulas_validation():
    print("VALIDATION INTÉGRALE DES FORMULES JAX-KEOPS")
    device = jax.devices()[0].platform
    print(f"Backend JAX détecté : {device}")

    key = jax.random.PRNGKey(0)
    
    # Dimensions réduites pour éviter le OOM sur l'implémentation JAX pure
    M, N, D = 100, 100, 5 
    
    X = jax.random.normal(key, (M, D), dtype=jnp.float32)
    Y = jax.random.normal(key, (N, D), dtype=jnp.float32)
    B = jax.random.normal(key, (N, 1), dtype=jnp.float32)

    # Lancement des tests pour chaque formule
    t_start_all = time.time()
    
    for formula_name, data in TEST_FORMULAS.items():
        # Compilation/Warm-up KeOps
        _ = jax_keops_convolution(formula_name, X, Y, B).block_until_ready()
        
        run_test_for_formula(formula_name, X, Y, B, data["ref_fn"])

    t_end_all = time.time()
    print("\n========================================================")
    print(f"TOUTES LES {len(TEST_FORMULAS)} FORMULES VÉRIFIÉES EN {t_end_all - t_start_all:.2f} secondes.")
    print("Intégration JAX-KeOps Validée Numériquement (Forward & Backward).")
    print("========================================================")


if __name__ == "__main__":
    test_all_formulas_validation()