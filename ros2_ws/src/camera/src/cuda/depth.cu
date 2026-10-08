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
                                                float disparity_scale,
                                                float min_depth_m,
                                                float max_depth_m) {
            const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) return;

            const auto* disparity_row = reinterpret_cast<const std::int16_t*>(
                reinterpret_cast<const std::uint8_t*>(disparity) + y * disparity_pitch);
            auto* depth_row = reinterpret_cast<float*>(
                reinterpret_cast<std::uint8_t*>(depth) + y * depth_pitch);

            const float d = static_cast<float>(disparity_row[x]) / disparity_scale;
            if (d <= 0.0F) {
                depth_row[x] = NAN;
                return;
            }

            const float depth_m = (fx_px * baseline_m) / d;
            if (!isfinite(depth_m) || depth_m < min_depth_m || depth_m > max_depth_m) {
                depth_row[x] = NAN;
                return;
            }
            depth_row[x] = depth_m;
        }
    }

    bool disparityToDepth(const CudaBuffer& disparity,
                          CudaBuffer& depth,
                          float fx_px,
                          float baseline_m,
                          float disparity_scale,
                          float min_depth_m,
                          float max_depth_m,
                          cudaStream_t stream) {
        if (!disparity.isAllocated() || !depth.isAllocated() || stream == nullptr) return false;

        if (!std::isfinite(fx_px) || fx_px <= 0.0F ||
            !std::isfinite(baseline_m) || baseline_m <= 0.0F ||
            !std::isfinite(disparity_scale) || disparity_scale <= 0.0F ||
            !std::isfinite(min_depth_m) || min_depth_m <= 0.0F ||
            !std::isfinite(max_depth_m) || max_depth_m <= min_depth_m) return false;

        if (disparity.width() != depth.width() ||
            disparity.height() != depth.height() ||
            disparity.channels() != 1 || depth.channels() != 1 ||
            disparity.elementSize() != sizeof(std::int16_t) ||
            depth.elementSize() != sizeof(float)) return false;

        const dim3 block(16, 16);
        const dim3 grid((disparity.width() + block.x - 1) / block.x,
                        (disparity.height() + block.y - 1) / block.y);

        disparityToDepthKernel<<<grid, block, 0, stream>>>(
            disparity.dataAs<std::int16_t>(), disparity.pitch(),
            depth.dataAs<float>(), depth.pitch(),
            disparity.width(), disparity.height(),
            fx_px, baseline_m, disparity_scale, min_depth_m, max_depth_m);

        return cudaGetLastError() == cudaSuccess;
    }
}
