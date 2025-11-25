#include <functional>
#include <numeric>
#include <utility>
#include <cmath> // Inclus pour être cohérent avec un fichier d'utilitaires

// Correction d'inclusion pour la version moderne de JAX/XLA
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// A helper function for extracting the relevant dimensions from `ffi::Buffer`s.
// In this example, we treat all leading dimensions as batch dimensions, so this
// function returns the total number of elements in the buffer, and the size of
// the last dimension.

template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    // Si c'est un scalaire sans dimension explicite, l'élément count est 1.
    return std::make_pair(buffer.element_count(), 1); 
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

// A wrapper function providing the interface between the XLA FFI call and our
// library function. This function handles the batch dimensions by calling 
// l'opération de carré dans une boucle.
ffi::Error squareImpl(ffi::Buffer<ffi::F32> x, 
                      ffi::ResultBuffer<ffi::F32> y) {
  
  // Utilisation de GetDims comme dans l'exemple RmsNorm
  auto [totalSize, lastDim] = GetDims(x);

  // Vérification de base (similaire à la vérification RmsNorm)
  if (totalSize == 0 || lastDim == 0) {
     // Nous ne devrions pas avoir besoin de vérifier les dimensions pour Square, 
     // mais pour suivre l'exemple RmsNorm nous incluons une vérification.
     return ffi::Error::InvalidArgument("Square input must not be empty.");
  }

  // Assurez-vous que les buffers d'entrée et de sortie ont la même taille.
  if (totalSize != y->element_count()) {
     return ffi::Error::InvalidArgument(
         "Input and output buffers must have the same total number of elements.");
  }

  // Accès direct aux données typées (float*)
  const float* x_data = x.typed_data();
  float* y_data = y->typed_data();

  // Boucle pour effectuer l'opération de carré sur tous les éléments du buffer.
  // Pour le carré, l'itération peut se faire sur totalSize.
  // Contrairement à RmsNorm, nous ne sautons pas par 'lastDim' ici, 
  // car nous traitons chaque élément individuellement.
  for (int64_t i = 0; i < totalSize; ++i) {
    y_data[i] = x_data[i] * x_data[i];
  }

  // Utilisation de ffi::Error::Success()
  return ffi::Error::Success();
}

// Wrap `SquareImpl` and specify the interface to XLA.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    square,           // Nom du symbole exposé (utilisé dans JAX)
    squareImpl,       // Fonction C++ réelle
    ffi::Ffi::Bind()
      // Note: RmsNorm utilise .Attr<float>("eps"), nous l'omettons ici.
      .Arg<ffi::Buffer<ffi::F32>>()   // Argument 0: x (Input Buffer float32)
      .Ret<ffi::Buffer<ffi::F32>>()   // Retour: y (Output Buffer float32)
);