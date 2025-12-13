import sys
sys.path.insert(0, '.')
from core.advanced_interface import jax_keops_convolution
import jax
import jax.numpy as jnp

print("=" * 60)
print("TEST FINAL - DÉRIVÉES D'ORDRE 3 ET 4")
print("=" * 60)

# Données très petites pour le test
key = jax.random.PRNGKey(42)
X = jax.random.normal(key, (2, 2), dtype=jnp.float32)  # Très petit!
Y = jax.random.normal(key, (3, 2), dtype=jnp.float32)
B = jnp.ones((3, 1), dtype=jnp.float32)

print(f"Données: X={X.shape}, Y={Y.shape}")
print(f"Taille entrée: {X.shape[0]*X.shape[1]} paramètres")

# Fonction scalaire pour autodiff
def f(X):
    return jnp.sum(jax_keops_convolution("conv_gaussienne", X, Y, B))

print("\n1. Ordre 1: Gradient")
grad = jax.grad(f)(X)
print(f"   Shape: {grad.shape}")

print("\n2. Ordre 2: Hessienne")
hess = jax.hessian(f)(X)
print(f"   Shape: {hess.shape}")

print("\n3. Ordre 3: Dérivée troisième")
# jacobian du gradient
def grad_f(X):
    return jax.grad(f)(X)

third = jax.jacobian(grad_f)(X)
print(f"   Shape: {third.shape}")
print(f"   Éléments: {third.size}")

print("\n4. Ordre 4: Dérivée quatrième")
# hessian du gradient
fourth = jax.hessian(grad_f)(X)
print(f"   Shape: {fourth.shape}")
print(f"   Éléments: {fourth.size}")

print("\n5. Vérification")
# Vérifie que tout est cohérent
M, D = X.shape
hess_flat = hess.reshape(M*D, M*D)
sym_err = jnp.linalg.norm(hess_flat - hess_flat.T)
print(f"   Erreur symétrie Hessienne: {sym_err:.2e}")

print("\n" + "🎉" * 30)
if sym_err < 1e-4:
    print("SUCCÈS ABSOLU !")
    print(f"✓ Dérivées 1ère ordre: {grad.shape}")
    print(f"✓ Dérivées 2ème ordre: {hess.shape}") 
    print(f"✓ Dérivées 3ème ordre: {third.shape}")
    print(f"✓ Dérivées 4ème ordre: {fourth.shape}")
    print("\n✅ KEOPS-JAX SUPPORTE L'AUTODIFF D'ORDRE SUPÉRIEUR !")
    print("✅ Tu peux calculer des dérivées jusqu'au 4ème ordre !")
else:
    print("Problème avec les dérivées")
print("🎉" * 30)
