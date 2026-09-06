#include <parallax/cuda/masked_depth_samples.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace parallax::cuda {

    namespace {
        __global__ void sampleMaskedDepthKernel(const std::uint8_t* mask,
                                                std::size_t mask_pitch,
                                                std::uint32_t mask_width,
                                                std::uint32_t mask_height,
                                                const float* depth,
                                                std::size_t depth_pitch,
                                                const float* map_x,
                                                std::size_t map_x_pitch,
                                                const float* map_y,
                                                std::size_t map_y_pitch,
                                                std::uint32_t width,
                                                std::uint32_t height,
                                                float fx,
                                                float fy,
                                                float cx,
                                                float cy,
                                                std::uint32_t stride,
                                                std::uint32_t max_samples,
                                                MaskedDepthPoint* samples,
                                                std::uint32_t* sample_count) {

            const std::uint32_t sample_x = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t sample_y = blockIdx.y * blockDim.y + threadIdx.y;

            const std::uint32_t x = sample_x * stride;
            const std::uint32_t y = sample_y * stride;

            if (x >= width || y >= height) return;

            const auto* map_x_row = reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(map_x) +
                                                                    static_cast<std::size_t>(y) * map_x_pitch);

            const auto* map_y_row = reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(map_y) +
                                                                    static_cast<std::size_t>(y) * map_y_pitch);

            const float source_x_f = map_x_row[x];
            const float source_y_f = map_y_row[x];

            if (!isfinite(source_x_f) || !isfinite(source_y_f)) return;

            const int source_x = static_cast<int>(lrintf(source_x_f));
            const int source_y = static_cast<int>(lrintf(source_y_f));

            if (source_x < 0 ||
                source_y < 0 ||
                source_x >= static_cast<int>(mask_width) ||
                source_y >= static_cast<int>(mask_height)) {

                return;
            }

            const auto* mask_row = reinterpret_cast<const std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(mask) +
                                                                         static_cast<std::size_t>(source_y) * mask_pitch);

            if (mask_row[source_x] == 0) return;

            const auto* depth_row = reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(depth) +
                                                                    static_cast<std::size_t>(y) * depth_pitch);

            const float z = depth_row[x];
            if (!isfinite(z) || z <= 0.0F) return;

            const std::uint32_t index = atomicAdd(sample_count, 1U);
            if (index >= max_samples) return;

            MaskedDepthPoint point{};
            point.x = (static_cast<float>(x) - cx) * z / fx;
            point.y = (static_cast<float>(y) - cy) * z / fy;
            point.z = z;

            samples[index] = point;
        }
    }

    bool sampleMaskedDepth(const std::uint8_t* mask,
                           std::size_t mask_pitch,
                           std::uint32_t mask_width,
                           std::uint32_t mask_height,
                           const CudaBuffer& depth,
                           const CudaBuffer& rectified_to_rgb_x,
                           const CudaBuffer& rectified_to_rgb_y,
                           float fx,
                           float fy,
                           float cx,
                           float cy,
                           std::uint32_t sample_stride,
                           std::uint32_t max_samples,
                           CudaBuffer& samples,
                           CudaBuffer& sample_count,
                           cudaStream_t stream) {

        if (mask == nullptr ||
            mask_pitch == 0 ||
            mask_width == 0 ||
            mask_height == 0 ||
            !depth.isAllocated() ||
            !rectified_to_rgb_x.isAllocated() ||
            !rectified_to_rgb_y.isAllocated() ||
            !samples.isAllocated() ||
            !sample_count.isAllocated() ||
            stream == nullptr ||
            sample_stride == 0 ||
            max_samples == 0 ||
            fx <= 0.0F ||
            fy <= 0.0F) {

            return false;
        }

        if (cudaMemsetAsync(sample_count.data(), 0, sizeof(std::uint32_t), stream) != cudaSuccess) {
            return false;
        }

        const std::uint32_t sample_width = (depth.width() + sample_stride - 1) / sample_stride;
        const std::uint32_t sample_height = (depth.height() + sample_stride - 1) / sample_stride;

        constexpr dim3 Threads{16, 16};

        const dim3 blocks{(sample_width + Threads.x - 1) / Threads.x,
                          (sample_height + Threads.y - 1) / Threads.y};

        sampleMaskedDepthKernel<<<blocks, Threads, 0, stream>>>(mask,
                                                                mask_pitch,
                                                                mask_width,
                                                                mask_height,
                                                                depth.dataAs<float>(),
                                                                depth.pitch(),
                                                                rectified_to_rgb_x.dataAs<float>(),
                                                                rectified_to_rgb_x.pitch(),
                                                                rectified_to_rgb_y.dataAs<float>(),
                                                                rectified_to_rgb_y.pitch(),
                                                                depth.width(),
                                                                depth.height(),
                                                                fx,
                                                                fy,
                                                                cx,
                                                                cy,
                                                                sample_stride,
                                                                max_samples,
                                                                samples.dataAs<MaskedDepthPoint>(),
                                                                sample_count.dataAs<std::uint32_t>());

        return cudaGetLastError() == cudaSuccess;
    }
}