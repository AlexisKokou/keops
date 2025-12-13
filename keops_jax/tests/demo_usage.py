# examples/demo_usage.py
"""
Exemple complet d'utilisation de KeOps-JAX.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from keops_jax import jax_keops_convolution, jax_keops_gaussian


def main():
    print("DEMO: KeOps-JAX - Convolutions géométriques avec autodiff complet")
    print("=" * 60)
    
    # 1. Configuration de base
    key = jax.random.PRNGKey(42)
    
    # 2. Données d'exemple
    # Points sources (à optimiser)
    M = 100
    # Points cibles (fixes)
    N = 200
    # Dimension
    D = 2
    
    X = jax.random.normal(key, (M, D)) * 0.5
    Y = jax.random.normal(key, (N, D))
    B = jnp.ones((N, 1))
    
    print(f"Exemple: {M} points sources, {N} points cibles, dimension {D}")
    
    # 3. Calcul simple avec différentes formules
    print("\n1. Calculs directs:")
    
    formulas = [
        ("conv_gaussienne", "Noyau Gaussien"),
        ("conv_cauchy", "Noyau de Cauchy"),
        ("mat_vec_mult", "Produit Matrice-Vecteur"),
    ]
    
    for formula_name, description in formulas:
        result = jax_keops_convolution(formula_name, X, Y, B)
        print(f"  {description}: shape {result.shape}, moyenne={jnp.mean(result):.3f}")
    
    # 4. Optimisation avec gradient
    print("\n2. Optimisation par gradient:")
    
    # Fonction de coût: on veut que X soit proche de Y selon le noyau gaussien
    def loss(X):
        # Négatif de la similarité (à minimiser)
        similarity = jax_keops_gaussian(X, Y, B)
        return -jnp.sum(similarity)
    
    # Gradient et optimisation simple
    grad_fn = jax.grad(loss)
    
    X_opt = X.copy()
    lr = 0.1
    
    print("  Descente de gradient...")
    for i in range(10):
        grad = grad_fn(X_opt)
        X_opt = X_opt - lr * grad
        current_loss = loss(X_opt)
        print(f"    Itération {i+1}: loss = {current_loss:.3f}")
    
    # 5. Utilisation de la Hessienne pour analyse
    print("\n3. Analyse avec la Hessienne:")
    
    def loss_for_hessian(X):
        return jnp.sum(jax_keops_gaussian(X[:10], Y[:10], B[:10]))
    
    # Calcul de la Hessienne (dérivées secondes)
    hessian_fn = jax.hessian(loss_for_hessian)
    H = hessian_fn(X[:10])
    
    print(f"  Hessienne calculée: shape {H.shape}")
    print(f"  Valeurs propres min: {jnp.linalg.eigvalsh(H).min():.3f}")
    print(f"  Valeurs propres max: {jnp.linalg.eigvalsh(H).max():.3f}")
    
    # 6. Visualisation
    print("\n4. Visualisation des résultats:")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Points initiaux
    axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.6, label='Sources')
    axes[0, 0].scatter(Y[:, 0], Y[:, 1], alpha=0.3, label='Cibles')
    axes[0, 0].set_title('Points initiaux')
    axes[0, 0].legend()
    axes[0, 0].axis('equal')
    
    # Points optimisés
    axes[0, 1].scatter(X_opt[:, 0], X_opt[:, 1], alpha=0.6, label='Sources opt.')
    axes[0, 1].scatter(Y[:, 0], Y[:, 1], alpha=0.3, label='Cibles')
    axes[0, 1].set_title('Après optimisation')
    axes[0, 1].legend()
    axes[0, 1].axis('equal')
    
    # Valeurs de la convolution
    result_gaussian = jax_keops_gaussian(X, Y, B)
    axes[1, 0].hist(result_gaussian.flatten(), bins=30, alpha=0.7)
    axes[1, 0].set_title('Distribution des valeurs (Gaussien)')
    axes[1, 0].set_xlabel('Valeur')
    axes[1, 0].set_ylabel('Fréquence')
    
    # Comparaison des formules
    results = []
    labels = []
    for formula_name, description in formulas:
        result = jax_keops_convolution(formula_name, X, Y, B)
        results.append(result.flatten())
        labels.append(description)
    
    axes[1, 1].boxplot([r.get() for r in results], labels=labels)
    axes[1, 1].set_title('Comparaison des formules')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print(f"  Figure sauvegardée: demo_results.png")
    
    # 7. Test de performance
    print("\n5. Test de performance sur grandes matrices:")
    
    # Test rapide
    X_large = jax.random.normal(key, (1000, 10))
    Y_large = jax.random.normal(key, (2000, 10))
    B_large = jnp.ones((2000, 1))
    
    import time
    t0 = time.time()
    result_large = jax_keops_gaussian(X_large, Y_large, B_large)
    result_large.block_until_ready()
    elapsed = time.time() - t0
    
    print(f"  Calcul avec 1000×2000×10: {elapsed:.2f}s")
    print(f"  Soit {(1000*2000*10)/elapsed/1e9:.2f} GOps/s")
    
    print("\n" + "=" * 60)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("=" * 60)
    print("Fonctionnalités démontrées:")
    print("1. Calculs de convolution avec différentes formules")
    print("2. Autodiff complet (gradients, Hessiennes)")
    print("3. Optimisation par gradient")
    print("4. Analyse avec dérivées secondes")
    print("5. Visualisation des résultats")
    print("6. Performance sur données de taille moyenne")


if __name__ == "__main__":
    main()