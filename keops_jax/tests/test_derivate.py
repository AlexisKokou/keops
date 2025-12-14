import time
import numpy as np
import jax
import jax.numpy as jnp
from keops_jax.core.jax_interface2 import jax_keops_forward, jax_keops_grad1, jax_keops_grad2

# Choisir le nom de la formule présent dans formulas.py
FORMULA = "conv_gaussienne"

def F_ref(X, Y, B):
    # X : (N,D), Y : (M,D), B : (M,1)
    # Calcule K(X,Y) @ B
    dist_sq = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)  # (N,M)
    K = jnp.exp(-dist_sq)
    return K @ B  # (N,1)

def test_derivatives():
    print("\n=== TEST KeOps ===")
    
    # --- Paramètres ---
    key = jax.random.PRNGKey(0)
    N, M, D = 20, 25, 3   # Garder N, M petits pour le Hessien de référence
    MAGNITUDE = 0.01      # Magnitude des vecteurs de test (G1, G2, v_full)
    
    # --- Génération des données ---
    key, key_X, key_Y, key_B, key_G1, key_G2, key_v_full = jax.random.split(key, 7)
    X = jax.random.normal(key_X, (N, D), dtype=jnp.float32)
    Y = jax.random.normal(key_Y, (M, D), dtype=jnp.float32)
    B = jax.random.normal(key_B, (M, 1), dtype=jnp.float32)

    # Référence forward (JAX)
    t0 = time.time()
    F_ref_val = F_ref(X, Y, B)
    jax.block_until_ready(F_ref_val)
    print("Forward (réf) terminé", (time.time() - t0)*1000, "ms")

    # Forward KeOps
    t0 = time.time()
    F_keops = jax_keops_forward(FORMULA, X, Y, B)
    jax.block_until_ready(F_keops)
    print("Forward KeOps terminé", (time.time() - t0)*1000, "ms")

    # Comparer forward
    diffF = float(jnp.linalg.norm(F_ref_val - F_keops))
    print("||F_ref - F_keops|| =", diffF)
    assert diffF < 1e-4, "Mismatch sur le forward"

    # --- Première dérivée (Grad 1) ---
    
    # Gradient de référence : on calcule le gradient de sum(F) par rapport à X
    def target_simple_sum(X_):
        return jnp.sum(F_ref(X_, Y, B))
    t0 = time.time()
    grad_ref = jax.grad(target_simple_sum)(X)
    jax.block_until_ready(grad_ref)
    print("Gradient (réf) terminé", (time.time() - t0)*1000, "ms")

    # Calcul de KeOps grad1 : G_up = gradient de la somme = vecteur de 1
    G_up = jnp.ones_like(F_keops) 
    t0 = time.time()
    dX_keops = jax_keops_grad1(FORMULA, X, Y, B, G_up)
    jax.block_until_ready(dX_keops)
    print("KeOps grad1 terminé", (time.time() - t0)*1000, "ms")

    # Comparer le gradient
    err_grad = float(jnp.linalg.norm(grad_ref - dX_keops))
    print("||grad_ref - dX_keops|| =", err_grad)
    assert err_grad < 1e-4, "Mismatch sur Grad1"

    # --- Deuxième dérivée (Grad 2 / Produit Hessien-Vecteur) ---
    print("\n--- Vérification Hessien ---")
    
    # 1. Génération des cotangents aléatoires de petite magnitude
    G1 = jax.random.normal(key_G1, (N,), dtype=jnp.float32) * MAGNITUDE
    G2 = jax.random.normal(key_G2, (N,), dtype=jnp.float32) * MAGNITUDE
    
    # 2. Référence JAX (Harmonisation Sémantique)
    # G1_full doit être de forme (N, 1) pour la multiplication
    G1_full = G1.reshape((N, 1))

    # Redéfinir la fonction cible dont on calcule la Hessienne: sum_i G1_i * F_i(X)
    def target_G1(X_):
        F_val = F_ref(X_, Y, B) # (N, 1)
        return jnp.sum(F_val * G1_full) # Somme pondérée par G1
        
    # La fonction gradient (le premier gradient)
    def grad_fn_G1(X_):
        return jax.grad(target_G1)(X_) # retourne d(sum(G1*F))/dX (N, D)

    # La direction pour jvp (second cotangent G2, forme (N, D))
    v_full_G2 = G2.reshape((N,1))
    v_full = jnp.tile(v_full_G2, (1, D)).astype(jnp.float32)
    
    # Calculer le produit Hessien-vecteur (Hv_ref)
    _, Hv_ref = jax.jvp(grad_fn_G1, (X,), (v_full,))
    jax.block_until_ready(Hv_ref)
    
    # 3. Calcul KeOps grad2
    t0 = time.time()
    d2X_keops = jax_keops_grad2(FORMULA, X, Y, B, G1, G2)
    jax.block_until_ready(d2X_keops)
    t_keops2 = (time.time() - t0) * 1000
    print("KeOps grad2 terminé", t_keops2, "ms")

    # 4. Comparaison
    diff2 = float(jnp.linalg.norm(Hv_ref - d2X_keops))
    norm_Hv_ref = float(jnp.linalg.norm(Hv_ref))
    
    print("||Hv_ref - d2X_keops|| (Erreur Absolue) =", diff2)
    print("||Hv_ref|| (Norme Référence) =", norm_Hv_ref)
        

    print("Terminé.")

if __name__ == "__main__":
    test_derivatives()