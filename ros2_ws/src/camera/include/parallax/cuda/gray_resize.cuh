#pragma once

#include <parallax/cuda/cuda_buffer.cuh>
#include <cuda_runtime.h>

namespace parallax::cuda {

bool downsampleGray2x(const CudaBuffer& input,
                      CudaBuffer& output,
                      cudaStream_t stream);

}  // namespace parallax::cuda
