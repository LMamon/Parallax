#pragma once

#include <parallax/cuda/cuda_buffer.cuh>
#include <cuda_runtime.h>

namespace parallax::cuda {

    bool disparityToDepth(const CudaBuffer& disparity,
                        CudaBuffer& depth,
                        float fx_px,
                        float baseline_m,
                        float disparity_scale,
                        cudaStream_t stream);
}