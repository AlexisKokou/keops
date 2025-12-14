# Correction de l'environnement JAX pour le sharding CPU
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4" # Utilise 4 cœurs virtuels

import jax
import jax.numpy as jnp
import numpy as np
import time


from keops_jax.core.jax_interface import jax_keops_convolution_vjp
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as PS

def test_sharding_scalability():
    print("   TEST DE SHARDING JAX (PARALLÉLISATION INTER-CŒURS)")
    
    # Configuration du Mesh (environnement sharded)
    devices = jax.local_devices()
    num_devices = len(devices)

    # L'erreur JAX indique que num_devices est 1, mais que le Mesh n'est pas actif.
    if num_devices < 1:
        print("Échec : Aucun périphérique détecté pour le sharding.")
        return

    # Créer un Mesh sur tous les périphériques détectés, avec un seul axe 'data'
    mesh_devices = create_device_mesh((num_devices,))
    mesh = Mesh(mesh_devices, axis_names=('data',))
    print(f"Mesh créé avec {num_devices} périphériques JAX.")
    
    # Définition des données
    key = jax.random.PRNGKey(42)
    M_big, N_big, D = 10_000 * num_devices, 10_000, 3 # M est mis à l'échelle pour le sharding
    
    X_big = jax.random.normal(key, (M_big, D), dtype=jnp.float32)
    Y_big = jax.random.normal(key, (N_big, D), dtype=jnp.float32) # Répliqué
    B_big = jax.random.normal(key, (N_big, 1), dtype=jnp.float32) # Répliqué

    # Définition du PjIT (Fonction de Compilation Statique et Sharded)

    # La fonction à sharder (elle utilise la primitive custom_vjp)
    def run_conv(X, Y, B):
        # L'ID de formule 0 est l'argument statique (nondiff_argnum)
        return jax_keops_convolution_vjp(0, X, Y, B)

    # 1. Spécification des partitions (PartitionSpec)
    # X est sharded sur l'axe 'data' (partitionné en M_big / num_devices)
    # Y et B sont répliqués (None)
    pspec_x = PS('data', None)
    pspec_y = PS(None, None)
    pspec_b = PS(None, None)
    
    # 2. Compilation PjIT (Forward)
    # Définir le JIT avec les règles de sharding (in_shardings, out_shardings)
    sharded_forward = jax.jit(run_conv, out_shardings=pspec_x, in_shardings=(pspec_x, pspec_y, pspec_b))

    print(f"\n1. Compilation PjIT (Forward) et Exécution du Sharding...")
    
    # --- CORRECTION CRITIQUE: Activation du mesh avec set_mesh ---
    # Le mesh doit être activé avant d'utiliser PartitionSpec
    jax.set_mesh(mesh)
    
    start = time.time()
    
    # Compilation JIT avec le mesh actif
    F_sharded = sharded_forward(X_big, Y_big, B_big)
    F_sharded.block_until_ready()
    
    time_sharded = (time.time() - start) * 1000
    
    print(f"Temps Sharded ({M_big} lignes): {time_sharded:.2f} ms")
    print(f"Forme de sortie après sharding : {F_sharded.shape}")

    # Test rapide de la backward pass sharded
    print("\n2. Test de la Backward Pass Sharded...")
    
    # Définir le JIT pour le gradient avec le mesh déjà actif
    grad_fn_sharded = jax.jit(jax.grad(lambda X_: jnp.sum(run_conv(X_, Y_big, B_big))),
                              out_shardings=pspec_x, in_shardings=pspec_x)
    
    start = time.time()
    G_sharded = grad_fn_sharded(X_big)
    G_sharded.block_until_ready()

    time_grad_sharded = (time.time() - start) * 1000
    
    print(f"Temps Backward Sharded : {time_grad_sharded:.2f} ms")
    print("Le Forward et le Backward fonctionnent dans un environnement sharded PjIT.")


if __name__ == "__main__":
    test_sharding_scalability()