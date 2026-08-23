#include <parallax/cuda/depth.cuh>

#include <cmath>
#include <cstdint>

namespace parallax::cuda {
    namespace {
        __global__ void disparityToDepthKernel(const std::int16_t* disparity,
                                                std::size_t disparity_pitch,
                                                float* depth,
                                                std::size_t depth_pitch,
                                                std::uint32_t width,
                                                std::uint32_t height,
                                                float fx_px, 
                                                float baseline_m,
                                                float disparity_scale) {

            const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height) { return; }

            const auto* disparity_row = reinterpret_cast<const std::int16_t*>(
                                        reinterpret_cast<const std::uint8_t*>(disparity) +
                                        y * disparity_pitch);

            auto* depth_row = reinterpret_cast<float*>(reinterpret_cast<std::uint8_t*>(depth) +
                                                        y * depth_pitch);

            const float d = static_cast<float>(disparity_row[x]) / disparity_scale;

            if (d <= 0.0f) {
                depth_row[x] = NAN;
                return;
            }

            depth_row[x] = (fx_px * baseline_m) / d;
        }
    }


    bool disparityToDepth(const CudaBuffer& disparity, 
                          CudaBuffer& depth, 
                          float fx_px, 
                          float baseline_m,
                          float disparity_scale,
                          cudaStream_t stream) {
    
        if (!disparity.isAllocated() || !depth.isAllocated() || stream == nullptr) {
            return false;
        }

        const dim3 block(16, 16);
        const dim3 grid((disparity.width() + block.x - 1) / block.x,
                        (disparity.height() + block.y - 1) / block.y);

        disparityToDepthKernel<<<grid, block, 0, stream>>>(disparity.dataAs<std::int16_t>(),
                                                            disparity.pitch(),
                                                            depth.dataAs<float>(),
                                                            depth.pitch(),
                                                            disparity.width(),
                                                            disparity.height(),
                                                            fx_px,
                                                            baseline_m,
                                                            disparity_scale);

        return cudaGetLastError() == cudaSuccess;
    }
}