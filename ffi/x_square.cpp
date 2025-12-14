#include <cmath>
#include <cstdint>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Retourne une paire (nombre total d'éléments, dernière dimension)
// - Si le buffer est scalaire (dims.size() == 0), on renvoie (0, 0)
// - Utile pour gérer des tenseurs avec une dimension "batch" + dernière dim utile
template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
    auto dims = buffer.dimensions();
    if (dims.size() == 0) {
        return std::make_pair(0, 0);
    }
    // element_count() = produit de toutes les dimensions
    // dims.back() = taille de la dernière dimension
    return std::make_pair(buffer.element_count(), dims.back());
}

// Calcul élémentaire x^2 pour un tableau contigu de float
// - size : nombre d'éléments à traiter
// - x : pointeur d'entrée
// - y : pointeur sortie (doit être alloué et au moins de taille 'size')
void ComputeXSquare(int64_t size, const float *x, float *y) {
    for (int64_t i = 0; i < size; ++i) {
        y[i] = x[i] * x[i];
    }
}

// Implémentation de la fonction FFI exposée
// - x : buffer d'entrée (F32)
// - y : buffer résultat (F32) déjà alloué par l'appelant
// Le comportement : applique ComputeXSquare sur la dernière dimension pour
// chaque élément du batch (si plusieurs dimensions)
ffi::Error XSquareImpl(ffi::Buffer<ffi::F32> x,
                                             ffi::ResultBuffer<ffi::F32> y) {
    auto [totalSize, lastDim] = GetDims(x);
    if (lastDim == 0) {
        // Entrée invalide (scalaire ou buffer vide)
        return ffi::Error::InvalidArgument("XSquare input must be an array");
    }

    // Traitement par "batch" : on parcourt le buffer par blocs de taille lastDim
    // Exemple : si tensor shape = [B, D], totalSize = B*D, lastDim = D
    for (int64_t n = 0; n < totalSize; n += lastDim) {
        // adresse du début du bloc courant dans x et y
        ComputeXSquare(lastDim, &(x.typed_data()[n]), &(y->typed_data()[n]));
    }

    return ffi::Error::Success();
}

// Déclaration de l'handler FFI exposé au runtime XLA
// - Nom : XSquare
// - Implémentation : XSquareImpl
// - Signature : Buffer<F32> -> Buffer<F32>
XLA_FFI_DEFINE_HANDLER_SYMBOL(
        XSquare, XSquareImpl,
        ffi::Ffi::Bind()
                .Arg<ffi::Buffer<ffi::F32>>()  // x
                .Ret<ffi::Buffer<ffi::F32>>()  // y
);
