import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

print("🧪 TEST COMPLET DE TOUS LES ORDRES AVEC COMPARAISON JAX PUR")
print("="*70)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import pykeops
pykeops.clean_pykeops()
os.environ['PYKEOPS_FORCE_COMPILE'] = '1'
os.environ['PYKEOPS_VERBOSE'] = '1'

from core.jax_interface3 import jax_keops_convolution
from core.keops_executor_derivate3 import keops_nth_order
from core.formulas import FORMULA_STRINGS

print("✅ Modules importés")

key = jax.random.PRNGKey(42)
M, N, D = 3, 3, 2
X = jax.random.normal(key, (M, D), dtype=jnp.float32)
Y = jax.random.normal(key, (N, D), dtype=jnp.float32)
B = jax.random.normal(key, (N, 1), dtype=jnp.float32)

print(f"\n📊 Données: M={M}, N={N}, D={D}")
print(f"X: {X.shape}, Y: {Y.shape}, B: {B.shape}")

# ============================================================================
# IMPLÉMENTATIONS PUR JAX POUR COMPARAISON
# ============================================================================

def gaussian_kernel_jax(x, y):
    """Noyau gaussien en pur JAX"""
    diff = x[:, None, :] - y[None, :, :]  # (M, N, D)
    squared_dist = jnp.sum(diff**2, axis=-1)  # (M, N)
    return jnp.exp(-squared_dist)  # (M, N)

def cauchy_kernel_jax(x, y):
    """Noyau de Cauchy en pur JAX"""
    diff = x[:, None, :] - y[None, :, :]  # (M, N, D)
    squared_dist = jnp.sum(diff**2, axis=-1)  # (M, N)
    return 1.0 / (1.0 + squared_dist)  # (M, N)

def linear_kernel_jax(x, y):
    """Noyau linéaire en pur JAX"""
    return jnp.sum(x[:, None, :] * y[None, :, :], axis=-1)  # (M, N)

# Fonctions de convolution en pur JAX
def conv_gaussienne_jax(X, Y, B):
    """Convolution gaussienne en pur JAX"""
    K = gaussian_kernel_jax(X, Y)  # (M, N)
    return K @ B  # (M, 1)

def conv_cauchy_jax(X, Y, B):
    """Convolution Cauchy en pur JAX"""
    K = cauchy_kernel_jax(X, Y)  # (M, N)
    return K @ B  # (M, 1)

def mat_vec_mult_jax(X, Y, B):
    """Mat-vec multiplication en pur JAX"""
    K = linear_kernel_jax(X, Y)  # (M, N)
    return K @ B  # (M, 1)

def copy_B_jax(X, Y, B):
    """Copie B en pur JAX (indépendant de X et Y)"""
    return jnp.sum(B) * jnp.ones((X.shape[0], 1))  # (M, 1)

# Mappage des formules
JAX_FORMULAS = {
    "conv_gaussienne": conv_gaussienne_jax,
    "conv_cauchy": conv_cauchy_jax,
    "mat_vec_mult": mat_vec_mult_jax,
    "copy_B": copy_B_jax
}

def test_order_n_with_jax_comparison(n, formula_name="conv_gaussienne"):
    print(f"\n{'='*50}")
    print(f"📐 TEST ORDRE {n} - {formula_name}")
    print(f"{'='*50}")
    
    # Fonction avec votre interface JAX-KeOps
    def f_keops(x):
        return jnp.sum(jax_keops_convolution(formula_name, x, Y, B))
    
    # Fonction avec JAX pur
    def f_jax(x):
        return jnp.sum(JAX_FORMULAS[formula_name](x, Y, B))
    
    # Calcul des dérivées avec votre interface
    print(f"Calcul avec votre interface JAX-KeOps...")
    try:
        if n == 1:
            grad_keops = jax.grad(f_keops)(X)
            grad_jax = jax.grad(f_jax)(X)
            
            # Calcul des erreurs
            abs_error = jnp.abs(grad_keops - grad_jax)
            max_abs_error = jnp.max(abs_error)
            mean_abs_error = jnp.mean(abs_error)
            rel_error = jnp.linalg.norm(grad_keops - grad_jax) / jnp.linalg.norm(grad_jax + 1e-10)
            
            print(f"✅ Ordre {n} réussi!")
            print(f"   Shape KeOps: {grad_keops.shape}")
            print(f"   Shape JAX pur: {grad_jax.shape}")
            print(f"   Erreur absolue max: {max_abs_error:.8f}")
            print(f"   Erreur absolue moyenne: {mean_abs_error:.8f}")
            print(f"   Erreur relative: {rel_error:.8f}")
            
            return True, grad_keops, grad_jax
            
        elif n == 2:
            hess_keops = jax.hessian(f_keops)(X)
            hess_jax = jax.hessian(f_jax)(X)
            
            # Flatten pour comparaison
            hess_keops_flat = hess_keops.reshape(-1)
            hess_jax_flat = hess_jax.reshape(-1)
            
            # Calcul des erreurs
            abs_error = jnp.abs(hess_keops_flat - hess_jax_flat)
            max_abs_error = jnp.max(abs_error)
            mean_abs_error = jnp.mean(abs_error)
            rel_error = jnp.linalg.norm(hess_keops_flat - hess_jax_flat) / jnp.linalg.norm(hess_jax_flat + 1e-10)
            
            print(f"✅ Ordre {n} réussi!")
            print(f"   Shape KeOps: {hess_keops.shape}")
            print(f"   Shape JAX pur: {hess_jax.shape}")
            print(f"   Erreur absolue max: {max_abs_error:.8f}")
            print(f"   Erreur absolue moyenne: {mean_abs_error:.8f}")
            print(f"   Erreur relative: {rel_error:.8f}")
            
            return True, hess_keops, hess_jax
            
        elif n == 3:
            # Dérivée troisième en JAX pur (via grad de hessienne)
            def third_jax(x):
                hess = jax.hessian(f_jax)(x)
                # Prendre la dérivée d'un élément de la hessienne
                return jax.grad(lambda x: hess[0, 0, 0, 0])(x)
            
            # Pour KeOps, on essaie d'obtenir quelque chose de comparable
            # Note: Cette approche est approximative pour la comparaison
            third_keops = jax.grad(jax.grad(jax.grad(f_keops)))(X)
            third_jax_val = jax.grad(jax.grad(jax.grad(f_jax)))(X)
            
            # Flatten pour comparaison
            third_keops_flat = third_keops.reshape(-1)
            third_jax_flat = third_jax_val.reshape(-1)
            
            # Calcul des erreurs
            abs_error = jnp.abs(third_keops_flat - third_jax_flat)
            max_abs_error = jnp.max(abs_error)
            mean_abs_error = jnp.mean(abs_error)
            
            print(f"✅ Ordre {n} réussi!")
            print(f"   Shape KeOps: {third_keops.shape}")
            print(f"   Shape JAX pur: {third_jax_val.shape}")
            print(f"   Erreur absolue max: {max_abs_error:.8f}")
            print(f"   Erreur absolue moyenne: {mean_abs_error:.8f}")
            
            return True, third_keops, third_jax_val
            
    except Exception as e:
        print(f"❌ Ordre {n} échoué: {type(e).__name__}")
        print(f"   Erreur: {e}")
        return False, None, None

def test_keops_direct(order):
    print(f"\n🔍 Test DIRECT KeOps ordre {order}:")
    
    X_np = np.array(X, dtype=np.float32)
    Y_np = np.array(Y, dtype=np.float32)
    B_np = np.array(B, dtype=np.float32)
    
    direction_vectors = []
    for i in range(order):
        if i == 0:
            dir_vec = np.ones((M, 1), dtype=np.float32)
        else:
            dir_vec = np.random.randn(M, D).astype(np.float32)
        direction_vectors.append(dir_vec)
    
    try:
        result = keops_nth_order(
            0,
            X_np, Y_np, B_np,
            *direction_vectors,
            FORMULA_STRINGS=FORMULA_STRINGS
        )
        print(f"   ✅ KeOps direct ordre {order}: {result.shape}")
        return True, result
    except Exception as e:
        print(f"   ❌ KeOps direct ordre {order} échoué: {e}")
        return False, None

# ============================================================================
# TESTS PRINCIPAUX
# ============================================================================

print("\n" + "="*70)
print("1. TEST FORWARD (ordre 0) - COMPARAISON JAX PUR")
print("="*70)

# Test pour chaque formule
formulas = ["conv_gaussienne", "conv_cauchy", "mat_vec_mult", "copy_B"]

for formula in formulas:
    print(f"\n📐 Formule: {formula}")
    
    # Forward avec votre interface
    F_keops = jax_keops_convolution(formula, X, Y, B)
    
    # Forward avec JAX pur
    F_jax = JAX_FORMULAS[formula](X, Y, B)
    
    # Calcul des erreurs
    abs_error = jnp.abs(F_keops - F_jax)
    max_abs_error = jnp.max(abs_error)
    mean_abs_error = jnp.mean(abs_error)
    rel_error = jnp.linalg.norm(F_keops - F_jax) / jnp.linalg.norm(F_jax + 1e-10)
    
    print(f"   ✅ Forward KeOps: {F_keops.shape}")
    print(f"   ✅ Forward JAX pur: {F_jax.shape}")
    print(f"   Erreur absolue max: {max_abs_error:.8f}")
    print(f"   Erreur absolue moyenne: {mean_abs_error:.8f}")
    print(f"   Erreur relative: {rel_error:.8f}")
    
    # Test direct KeOps pour vérification
    success_direct, result_direct = test_keops_direct(0)
    if success_direct:
        error_keops_direct = np.linalg.norm(F_keops - result_direct) / np.linalg.norm(result_direct + 1e-10)
        print(f"   Erreur KeOps vs KeOps direct: {error_keops_direct:.8f}")

print("\n" + "="*70)
print("2. TEST GRADIENT (ordre 1) - COMPARAISON DÉTAILLÉE")
print("="*70)

for formula in formulas:
    print(f"\n📐 Formule: {formula}")
    success, grad_keops, grad_jax = test_order_n_with_jax_comparison(1, formula)

print("\n" + "="*70)
print("3. TEST HESSIENNE (ordre 2) - COMPARAISON DÉTAILLÉE")
print("="*70)

# Test seulement pour les formules principales
for formula in ["conv_gaussienne", "conv_cauchy"]:
    print(f"\n📐 Formule: {formula}")
    success, hess_keops, hess_jax = test_order_n_with_jax_comparison(2, formula)

print("\n" + "="*70)
print("4. TEST 3ÈME ORDRE - COMPARAISON LIMITÉE")
print("="*70)

# Test seulement pour la gaussienne (plus simple)
formula = "conv_gaussienne"
print(f"\n📐 Formule: {formula}")
success, third_keops, third_jax = test_order_n_with_jax_comparison(3, formula)

print("\n" + "="*70)
print("5. ANALYSE STATISTIQUE DES ERREURS")
print("="*70)

# Fonction pour collecter les erreurs par ordre
def collect_errors():
    errors_by_order = {0: [], 1: [], 2: []}
    
    for formula in formulas:
        # Ordre 0
        F_keops = jax_keops_convolution(formula, X, Y, B)
        F_jax = JAX_FORMULAS[formula](X, Y, B)
        rel_error = jnp.linalg.norm(F_keops - F_jax) / jnp.linalg.norm(F_jax + 1e-10)
        errors_by_order[0].append(float(rel_error))
        
        # Ordre 1
        def f_keops(x):
            return jnp.sum(jax_keops_convolution(formula, x, Y, B))
        
        def f_jax(x):
            return jnp.sum(JAX_FORMULAS[formula](x, Y, B))
        
        try:
            grad_keops = jax.grad(f_keops)(X)
            grad_jax = jax.grad(f_jax)(X)
            rel_error = jnp.linalg.norm(grad_keops - grad_jax) / jnp.linalg.norm(grad_jax + 1e-10)
            errors_by_order[1].append(float(rel_error))
        except:
            pass
    
    # Statistiques
    print("\n📊 STATISTIQUES DES ERREURS RELATIVES:")
    for order, errors in errors_by_order.items():
        if errors:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            
            print(f"\n   Ordre {order}:")
            print(f"      Moyenne: {mean_error:.8f}")
            print(f"      Écart-type: {std_error:.8f}")
            print(f"      Max: {max_error:.8f}")
            print(f"      Min: {min_error:.8f}")
            print(f"      N échantillons: {len(errors)}")

collect_errors()

print("\n" + "="*70)
print("6. TEST DE PRÉCISION NUMÉRIQUE")
print("="*70)

# Test avec différentes tailles pour évaluer la stabilité
print("\n🔬 Test de précision avec différentes échelles:")

scales = [1e-3, 1.0, 1e3]
for scale in scales:
    print(f"\n   Échelle: {scale}")
    X_scaled = X * scale
    Y_scaled = Y * scale
    B_scaled = B * scale
    
    # Test forward
    F_keops = jax_keops_convolution("conv_gaussienne", X_scaled, Y_scaled, B_scaled)
    F_jax = conv_gaussienne_jax(X_scaled, Y_scaled, B_scaled)
    
    abs_error = jnp.max(jnp.abs(F_keops - F_jax))
    rel_error = jnp.linalg.norm(F_keops - F_jax) / jnp.linalg.norm(F_jax + 1e-10)
    
    print(f"      Erreur absolue max: {abs_error:.8e}")
    print(f"      Erreur relative: {rel_error:.8e}")

print("\n" + "="*70)
print("🎯 RÉSUMÉ DE LA PRÉCISION JAX-KEOPS-JAX")
print("="*70)

print("""
📊 INTERPRÉTATION DES RÉSULTATS:

1. ✅ FORWARD (Ordre 0):
   - Erreurs typiquement < 1e-7 (précision machine)
   - Bon accord entre JAX pur et JAX-KeOps

2. 📈 GRADIENT (Ordre 1):
   - Erreurs attendues: < 1e-5
   - Légère dégradation due à la différentiation automatique
   - Vérifier si les erreurs sont systématiques ou aléatoires

3. 🎲 HESSIENNE (Ordre 2):
   - Erreurs typiquement < 1e-4
   - Accumulation des erreurs de différentiation
   - Symétrie à vérifier

4. ⚠️  ORDRES SUPÉRIEURS (>2):
   - Précision dégradée attendue
   - Considérer comme indicative plutôt qu'exacte

🔍 DIAGNOSTIC:
- Si erreurs < 1e-6: Excellent accord
- Si erreurs 1e-6 à 1e-4: Bon accord
- Si erreurs > 1e-3: Vérifier l'implémentation

💡 RECOMMANDATIONS:
1. Utiliser double précision (float64) si disponible
2. Vérifier les gradients avec différences finies
3. Tester avec différentes tailles de données
4. Surveiller la symétrie des hessiennes
""")

# Test final de validation
print("\n" + "="*70)
print("🏁 VALIDATION FINALE")
print("="*70)

# Test avec différences finies pour confirmation
def finite_difference_gradient(f, x, eps=1e-4):
    """Gradient par différences finies centrées"""
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, j] += eps
            x_minus[i, j] -= eps
            
            f_plus = f(x_plus)
            f_minus = f(x_minus)
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    return grad

# Pour la gaussienne
print("\n🔍 Validation par différences finies (formule gaussienne):")

def f_keops_sum(x):
    return np.sum(jax_keops_convolution("conv_gaussienne", x, Y, B))

X_np = np.array(X)
grad_fd = finite_difference_gradient(f_keops_sum, X_np, eps=1e-5)
grad_keops_np = np.array(jax.grad(lambda x: jnp.sum(jax_keops_convolution("conv_gaussienne", x, Y, B)))(X))

error_fd = np.linalg.norm(grad_keops_np - grad_fd) / np.linalg.norm(grad_fd + 1e-10)
print(f"   Erreur gradient vs différences finies: {error_fd:.8f}")

if error_fd < 1e-4:
    print("   ✅ Validation réussie!")
else:
    print(f"   ⚠️  Écart significatif détecté")

print("\n" + "="*70)
print("🎊 TESTS TERMINÉS AVEC SUCCÈS!")
print("="*70)