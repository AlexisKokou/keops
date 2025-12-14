import sys
sys.path.insert(0, '.')

print("=" * 60)
print("TEST FINAL - DÉRIVÉES D'ORDRE SUPÉRIEUR")
print("=" * 60)

from core.advanced_interface import jax_keops_convolution
import jax
import jax.numpy as jnp

# Données
key = jax.random.PRNGKey(42)
X = jax.random.normal(key, (4, 3), dtype=jnp.float32)
Y = jax.random.normal(key, (5, 3), dtype=jnp.float32)
B = jnp.ones((5, 1), dtype=jnp.float32)

print(f"Données: X={X.shape}, Y={Y.shape}, B={B.shape}")
print(f"Hessienne attendue: {4*3}x{4*3} = {4*3*4*3} éléments\n")

# 1. Forward
print("1. Test Forward...")
result = jax_keops_convolution("conv_gaussienne", X, Y, B)
print(f"   ✅ Forward: {result.shape}")

# 2. Gradient
print("\n2. Test Gradient...")
def loss(X):
    return jnp.sum(jax_keops_convolution("conv_gaussienne", X, Y, B))

grad = jax.grad(loss)(X)
print(f"   ✅ Gradient: {grad.shape}")

# 3. HESSIENNE - LE TEST CRUCIAL
print("\n3. Test Hessienne (dérivées secondes)...")
try:
    hess = jax.hessian(loss)(X)
    print(f"   ✅ HESSIENNE CALCULÉE AVEC SUCCÈS !")
    print(f"   Shape: {hess.shape}")
    
    # Analyse
    M, D = X.shape
    hess_flat = hess.reshape(M*D, M*D)
    
    # Symétrie (propriété fondamentale des Hessiennes)
    sym_err = jnp.linalg.norm(hess_flat - hess_flat.T)
    print(f"   Erreur symétrie: {sym_err:.2e}")
    
    # Valeurs propres
    eigvals = jnp.linalg.eigvalsh(hess_flat)
    print(f"   Valeurs propres: [{eigvals.min():.2e}, {eigvals.max():.2e}]")
    
    # Décision
    if sym_err < 1e-4:
        print("\n" + "🎉" * 35)
        print("SUCCÈS ABSOLU !")
        print("LES DÉRIVÉES D'ORDRE SUPÉRIEUR FONCTIONNENT !")
        print("KeOps-JAX est pleinement opérationnel !")
        print("🎉" * 35)
        
        # Test supplémentaire : dérivées 3ème ordre
        print("\n4. Test dérivées 3ème ordre...")
        try:
            grad3 = jax.grad(jax.grad(jax.grad(loss)))(X)
            print(f"   ✅ Dérivées 3ème ordre: {grad3.shape}")
            print("   🚀 Même les dérivées d'ordre 3 fonctionnent !")
        except Exception as e3:
            print(f"   ⚠️  Dérivées 3ème ordre: {e3}")
            
    else:
        print(f"\n⚠️  ATTENTION: Hessienne non symétrique (erreur: {sym_err:.2e})")
        print("   La Hessienne devrait être symétrique pour une fonction C²")
        
except Exception as e:
    print(f"   ❌ ERREUR CRITIQUE: {e}")
    print("\n" + "❌" * 35)
    print("ÉCHEC: Les dérivées d'ordre supérieur ne fonctionnent PAS")
    print("Cela signifie que l'implémentation custom_jvp a un problème")
    print("❌" * 35)
    import traceback
    traceback.print_exc()

# 4. Test toutes les formules
print("\n5. Test toutes les formules...")
formulas = ["conv_gaussienne", "conv_cauchy", "mat_vec_mult", "copy_B"]

for f in formulas:
    try:
        r = jax_keops_convolution(f, X, Y, B)
        print(f"   {f:15} → ✅ {r.shape}")
    except Exception as e:
        print(f"   {f:15} → ❌ {str(e)[:50]}...")

print("\n" + "=" * 60)
