#include <cmath>
#include <cstdint>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();
  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}


void ComputeXSquare(int64_t size, const float *x, float *y) {
  for (int64_t i = 0; i < size; ++i) {
    y[i] = x[i] * x[i];
  }
}

ffi::Error XSquareImpl(ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y) {
  auto [totalSize, lastDim] = GetDims(x);
  if (lastDim == 0) {
    return ffi::Error::InvalidArgument("XSquare input must be an array");
  }

  // Batch processing (identique à doc)
  for (int64_t n = 0; n < totalSize; n += lastDim) {
    ComputeXSquare(lastDim, &(x.typed_data()[n]), &(y->typed_data()[n]));
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    XSquare, XSquareImpl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Ret<ffi::Buffer<ffi::F32>>()  // y
);
