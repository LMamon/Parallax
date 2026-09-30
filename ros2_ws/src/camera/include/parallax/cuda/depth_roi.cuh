#pragma once

#include <parallax/cuda/cuda_buffer.cuh>

#include <cuda_runtime.h>

#include <cstdint>

namespace parallax::cuda {

    struct DepthRoiRequest {
        std::int32_t center_x = 0;
        std::int32_t center_y = 0;
    };

    struct DepthRoiResult {
        float depth_m = 0.0F;
        std::uint32_t valid_samples = 0;
        std::uint32_t sampled_pixels = 0;
    };

    bool reduceDepthRois(const CudaBuffer& depth,
                        const CudaBuffer& requests,
                        CudaBuffer& results,
                        std::uint32_t request_count,
                        std::uint32_t radius,
                        cudaStream_t stream);
}