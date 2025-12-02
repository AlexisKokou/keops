import jax
import jax.numpy as jnp
from .keops_executor import keops_forward, keops_backward


@jax.custom_vjp
def jax_keops_convolution(formula_id, X, Y, B):
    """
    Point d'entrée JAX pour appeler une implémentation KeOps extérieure
    via des callbacks "purs" (pure_callback). Ce wrapper est enregistré
    avec une VJP personnalisée pour permettre la différentiation automatique.

    Parameters
    ----------
    formula_id : int32
        Identifiant du kernel/forme à utiliser (ex: 0 = Gaussian).
    X, Y, B : JAX arrays
        Entrées du calcul KeOps. On suppose X de forme (M, D).
    Returns
    -------
    out : array
        Résultat de la convolution KeOps de forme (M, 1) et dtype = X.dtype.
    """
    M = X.shape[0]
    # Spécifie la forme et le dtype attendus par le callback pur.
    out_shape = jax.ShapeDtypeStruct((M, 1), X.dtype)

    # Callback "purs" exécuté hors du tracer JAX : appelle la routine keops_forward.
    # Le cast int() garantit un type python natif pour la pure_callback.
    def fwd_callback(formula_id_, X_, Y_, B_):
        return keops_forward(int(formula_id_), X_, Y_, B_)

    # Appel non traçable (pure_callback) : permet d'exécuter du code externe/impur
    # tout en intégrant le résultat dans le graphe JAX.
    return jax.pure_callback(fwd_callback, out_shape, formula_id, X, Y, B)


# On définit ici la passe avant (fwd) et la passe arrière (bwd) utilisées
# par JAX pour la différentiation. Le schéma est :
# fwd(...) -> out, residuals
# bwd(residuals, g) -> cotangents (gradients) pour chaque argument de la primitif


def fwd(formula_id, X, Y, B):
    """
    Passe avant pour la VJP : on réutilise le wrapper jax_keops_convolution
    pour calculer la sortie et on stocke les entrants comme résiduels nécessaires
    pour la rétropropagation.
    """
    out = jax_keops_convolution(formula_id, X, Y, B)
    # On retourne aussi les valeurs nécessaires pour le bwd (formula_id, X, Y, B).
    return out, (formula_id, X, Y, B)


def bwd(res, G):
    """
    Passe arrière : calcule la VJP (produit vecteur-jacobien) en appelant
    la routine keops_backward via un pure_callback. JAX attend que bwd
    retourne un tuple de cotangentes correspondant aux arguments originaux
    de la primitive.
    Parameters
    ----------
    res : tuple
        Résidus retournés par fwd (formula_id, X, Y, B).
    G : array
        Gradient de la sortie (cotangente) de forme (M, 1).
    Returns
    -------
    tuple
        Cotangentes (dX, None, None, None) : on ne fournit de gradient que
        pour X ici ; les autres arguments (formula_id, Y, B) restent None.
    """
    formula_id, X, Y, B = res
    M, D = X.shape

    # Déclaration de la forme attendue du gradient retourné pour dX.
    dx_shape = jax.ShapeDtypeStruct((M, D), X.dtype)

    # Callback qui appelle la routine keops_backward (implémentation externe)
    # et retourne le gradient par rapport à X.
    def bwd_callback(formula_id_, X_, Y_, B_, G_):
        return keops_backward(int(formula_id_), X_, Y_, B_, G_)

    # pure_callback pour exécuter l'implémentation externe et récupérer dX.
    dX = jax.pure_callback(
        bwd_callback,
        dx_shape,
        formula_id, X, Y, B, G
    )

    # On retourne un tuple de la même longueur que les arguments de jax_keops_convolution.
    # Ici seuls X a un gradient, les autres sont None.
    return dX, None, None, None


# Register VJP
# On associe explicitement les fonctions fwd et bwd à la primitive custom_vjp.
jax_keops_convolution.defvjp(fwd, bwd)
