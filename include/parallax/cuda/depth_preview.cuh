#pragma once
#include <parallax/cuda/cuda_buffer.cuh>
#include <cstdint>
namespace parallax::cuda {
    bool downsampleDepthNearest(const CudaBuffer& source, CudaBuffer& destination,
                                std::uint32_t stride, cudaStream_t stream);
}
