# keops_jax/core/keops_executor_nth_order.py
"""Interface KeOps pour les dérivées d'ordre n - CORRIGÉ"""

import numpy as np
from pykeops.numpy import Genred

_KERNEL_CACHE = {}

def make_nth_order_kernel(formula_string, D, order):
    """Crée un kernel KeOps pour la dérivée d'ordre n"""
    cache_key = (formula_string, D, order)
    if cache_key in _KERNEL_CACHE:
        return _KERNEL_CACHE[cache_key]
    
    if order == 0:
        aliases = [f"X = Vi({D})", f"Y = Vj({D})", "B = Vj(1)"]
        kernel_str = formula_string
    else:
        current_formula = formula_string
        aliases = [f"X = Vi({D})", f"Y = Vj({D})", "B = Vj(1)"]
        
        # Pour les gradients, ajouter les directions
        for i in range(1, order + 1):
            current_formula = f"Grad({current_formula}, X, V{i})"
            aliases.append(f"V{i} = Vi({D})")
        
        kernel_str = current_formula
    
    kernel = Genred(kernel_str, aliases, reduction_op="Sum", axis=1)
    _KERNEL_CACHE[cache_key] = kernel
    return kernel

def keops_nth_order(formula_id, X_np, Y_np, B_np, formula_strings, *direction_vectors):
    """Calcule la dérivée d'ordre n via KeOps avec formules corrigées"""
    X_np = np.asarray(X_np, dtype=np.float32)
    Y_np = np.asarray(Y_np, dtype=np.float32)
    B_np = np.asarray(B_np, dtype=np.float32)
    direction_arrays = [np.asarray(v, dtype=np.float32) for v in direction_vectors]
    
    M, D = X_np.shape
    order = len(direction_arrays)
    
    formula = formula_strings[formula_id]
    
    # Création du kernel avec syntaxe KeOps correcte
    aliases = [f"X = Vi({D})", f"Y = Vj({D})", "B = Vj(1)"]
    
    if order == 0:
        # Forward direct
        kernel = Genred(formula, aliases, reduction_op='Sum', axis=1)
        result = kernel(X_np, Y_np, B_np)
    else:
        # Construction des dérivées avec Grad
        current_formula = formula
        current_aliases = aliases.copy()
        
        for i, direction in enumerate(direction_arrays):
            var_name = f"V{i}"
            current_aliases.append(f"{var_name} = Vi({D})")
            current_formula = f"Grad({current_formula}, X, {var_name})"
        
        kernel = Genred(current_formula, current_aliases, reduction_op='Sum', axis=1)
        args = [X_np, Y_np, B_np] + direction_arrays
        result = kernel(*args)
    
    return result

def keops_gradient_vector(formula_id, X_np, Y_np, B_np, formula_strings):
    """Retourne le gradient vectoriel (M, D) en calculant séparément chaque composante"""
    X_np = np.asarray(X_np, dtype=np.float32)
    Y_np = np.asarray(Y_np, dtype=np.float32)
    B_np = np.asarray(B_np, dtype=np.float32)
    
    N, D = X_np.shape
    formula = formula_strings[formula_id]
    
    # Calculer les composantes du gradient séparément
    gradient_components = []
    
    for d in range(D):
        # Vecteur unitaire pour la dimension d
        direction = np.zeros((N, D), dtype=np.float32)
        direction[:, d] = 1.0
        
        # Calculer la dérivée directionnelle pour cette composante
        kernel = make_nth_order_kernel(formula, D, 1)
        component = kernel(X_np, Y_np, B_np, direction)
        
        # component a la forme (N, 1), on veut juste la première colonne
        gradient_components.append(component[:, 0:1])  # Garder dimension (N, 1)
    
    # Concaténer toutes les composantes pour obtenir (N, D)
    gradient_vector = np.concatenate(gradient_components, axis=1)
    return gradient_vector

def keops_directional_derivative(formula_id, X_np, Y_np, B_np, direction, formula_strings):
    """Retourne la dérivée directionnelle (M, 1) = gradient · direction"""
    X_np = np.asarray(X_np, dtype=np.float32)
    Y_np = np.asarray(Y_np, dtype=np.float32)
    B_np = np.asarray(B_np, dtype=np.float32)
    direction_np = np.asarray(direction, dtype=np.float32)
    
    N, D = X_np.shape
    formula = formula_strings[formula_id]
    
    # Pour la dérivée directionnelle, on utilise le kernel comme avant
    kernel = make_nth_order_kernel(formula, D, 1)
    result = kernel(X_np, Y_np, B_np, direction_np)
    
    # Si le résultat a la forme (N, D), on calcule le produit scalaire avec la direction
    if result.shape[1] == D:
        # result est le gradient complet, on fait le produit scalaire
        directional_deriv = np.sum(result * direction_np, axis=1, keepdims=True)
        return directional_deriv
    else:
        # Le résultat est déjà une dérivée directionnelle
        return result