#include <parallax/cuda/depth_preview.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace parallax::cuda {
    namespace {
        constexpr std::uint32_t MaxPreviewStride = 8;
        constexpr std::uint32_t MaxPreviewSamples = MaxPreviewStride * MaxPreviewStride;

        __device__ void sortSmall(float* values, std::uint32_t count) {
            for (std::uint32_t i = 1; i < count; ++i) {
                const float value = values[i];
                std::uint32_t j = i;

                while (j > 0 && values[j - 1] > value) {
                    values[j] = values[j - 1];
                    --j;
                }

                values[j] = value;
            }
        }

        __global__ void downsampleDepthRobustKernel(const float* source,
                                                    std::size_t source_pitch,
                                                    std::uint32_t source_width,
                                                    std::uint32_t source_height,
                                                    float* destination,
                                                    std::size_t destination_pitch,
                                                    std::uint32_t destination_width,
                                                    std::uint32_t destination_height,
                                                    std::uint32_t stride) {

            const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= destination_width || y >= destination_height) return;

            float samples[MaxPreviewSamples];
            std::uint32_t count = 0;

            const std::uint32_t source_x = x * stride;
            const std::uint32_t source_y = y * stride;

            for (std::uint32_t dy = 0; dy < stride; ++dy) {
                const std::uint32_t sy = source_y + dy;
                if (sy >= source_height) break;

                const auto* source_row = reinterpret_cast<const float*>(
                    reinterpret_cast<const std::uint8_t*>(source) +
                    static_cast<std::size_t>(sy) * source_pitch);

                for (std::uint32_t dx = 0; dx < stride; ++dx) {
                    const std::uint32_t sx = source_x + dx;
                    if (sx >= source_width) break;

                    const float depth_m = source_row[sx];
                    if (isfinite(depth_m) && depth_m > 0.0F) {
                        samples[count++] = depth_m;
                    }
                }
            }

            auto* destination_row = reinterpret_cast<float*>(
                reinterpret_cast<std::uint8_t*>(destination) +
                static_cast<std::size_t>(y) * destination_pitch);

            if (count < 2) {
                destination_row[x] = NAN;
                return;
            }

            sortSmall(samples, count);

            const float median = samples[count / 2U];
            const float tolerance_m = fmaxf(0.05F, median * 0.05F);

            float sum = 0.0F;
            std::uint32_t support = 0;

            for (std::uint32_t i = 0; i < count; ++i) {
                if (fabsf(samples[i] - median) <= tolerance_m) {
                    sum += samples[i];
                    ++support;
                }
            }

            destination_row[x] = support >= 2
                ? sum / static_cast<float>(support)
                : NAN;
        }
    }

    bool downsampleDepthRobust(const CudaBuffer& source,
                               CudaBuffer& destination,
                               std::uint32_t stride,
                               cudaStream_t stream) {

        if (!source.isAllocated() || !destination.isAllocated() ||
            stream == nullptr || stride == 0 || stride > MaxPreviewStride) {
            return false;
        }

        if (source.channels() != 1 || destination.channels() != 1 ||
            source.elementSize() != sizeof(float) ||
            destination.elementSize() != sizeof(float)) {
            return false;
        }

        const std::uint32_t expected_width = (source.width() + stride - 1U) / stride;
        const std::uint32_t expected_height = (source.height() + stride - 1U) / stride;

        if (destination.width() != expected_width ||
            destination.height() != expected_height) {
            return false;
        }

        const dim3 block(16, 16);
        const dim3 grid((destination.width() + block.x - 1) / block.x,
                        (destination.height() + block.y - 1) / block.y);

        downsampleDepthRobustKernel<<<grid, block, 0, stream>>>(
            source.dataAs<float>(),
            source.pitch(),
            source.width(),
            source.height(),
            destination.dataAs<float>(),
            destination.pitch(),
            destination.width(),
            destination.height(),
            stride);

        return cudaGetLastError() == cudaSuccess;
    }
}
