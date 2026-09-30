#include <parallax/cuda/depth_roi.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace parallax::cuda {

    namespace {
        constexpr std::uint32_t kMaxRadius = 3;
        constexpr std::uint32_t kMaxSamples = (kMaxRadius * 2 + 1) * (kMaxRadius * 2 + 1);

        /*
         * Object3D only needs one representative depth per semantic ROI.
         * One block handles one ROI and thread 0 performs the tiny bounded
         * reduction. The work is intentionally serial inside the block because
         * the maximum input is only 49 samples; avoiding a more complicated
         * reduction keeps launch/setup overhead and implementation complexity low.
         */
        __global__ void reduceDepthRoisKernel(const float* depth,
                                              std::size_t depth_pitch,
                                              std::uint32_t width,
                                              std::uint32_t height,
                                              const DepthRoiRequest* requests,
                                              DepthRoiResult* results,
                                              std::uint32_t request_count,
                                              std::uint32_t radius) {

            const std::uint32_t index = blockIdx.x;

            if (index >= request_count || threadIdx.x != 0) return;

            const auto request = requests[index];

            float samples[kMaxSamples];
            std::uint32_t valid_count = 0;
            std::uint32_t sampled_count = 0;

            // Clipping the ROI to the actual depth image allows Border detections
            // to remain usable without reading outside the allocation.
            for (int dy = -static_cast<int>(radius); dy <= static_cast<int>(radius); ++dy) {

                const int y = request.center_y + dy;
                if (y < 0 || y >= static_cast<int>(height)) continue;

                const auto* row = reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(depth) +
                                                                 static_cast<std::size_t>(y) * depth_pitch);

                for (int dx = -static_cast<int>(radius); dx <= static_cast<int>(radius); ++dx) {

                    const int x = request.center_x + dx;

                    if (x < 0 || x >= static_cast<int>(width)) continue;
                    ++sampled_count;

                    const float value = row[x];

                    if (!isfinite(value) || value <= 0.0F) continue;

                    samples[valid_count++] = value;
                }
            }

            DepthRoiResult result{};
            result.valid_samples = valid_count;
            result.sampled_pixels = sampled_count;

            if (valid_count == 0) {
                results[index] = result;
                return;
            }

            /*
             * At most 49 values are present. Insertion sort is deterministic,
             * bounded, and simpler than introducing a general GPU sort just to
             * compute a small median.
             */
            for (std::uint32_t i = 1; i < valid_count; ++i) {
                const float value = samples[i];
                std::uint32_t j = i;

                while (j > 0 && samples[j - 1] > value) {
                    samples[j] = samples[j - 1];
                    --j;
                }

                samples[j] = value;
            }

            if ((valid_count & 1U) != 0U) {
                result.depth_m = samples[valid_count / 2];
            } else {
                const auto upper = valid_count / 2;
                result.depth_m = 0.5F * (samples[upper - 1] + samples[upper]);
            }

            results[index] = result;
        }

    }

    bool reduceDepthRois(const CudaBuffer& depth,
                         const CudaBuffer& requests,
                         CudaBuffer& results,
                         std::uint32_t request_count,
                         std::uint32_t radius,
                         cudaStream_t stream) {

        if (!depth.isAllocated() ||
            !requests.isAllocated() ||
            !results.isAllocated() ||
            stream == nullptr ||
            request_count == 0 ||
            radius > kMaxRadius) {
            return false;
        }

        reduceDepthRoisKernel<<<request_count, 1, 0, stream>>>(depth.dataAs<float>(),
                                                               depth.pitch(),
                                                               depth.width(),
                                                               depth.height(),
                                                               requests.dataAs<DepthRoiRequest>(),
                                                               results.dataAs<DepthRoiResult>(),
                                                               request_count,
                                                               radius);

        return cudaGetLastError() == cudaSuccess;
    }

}