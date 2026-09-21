#pragma once

#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/isp/frame_types.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace parallax::isp {

    struct CanonicalRgbParameters {
        float white_balance[3]{1.0F, 1.0F, 1.0F};
        
        float color_matrix[9]{
            1.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 1.0F,
        };

        float inverse_gamma = 1.0F / 2.2F;
        float linear_white_level = 959.0F;
    };

    struct DeviceIspStatistics {
        std::uint32_t luminance_histogram[256]{};

        unsigned long long red_sum = 0;
        unsigned long long green_sum = 0;
        unsigned long long blue_sum = 0;
        unsigned long long color_samples = 0;
        unsigned long long total_samples = 0;
        unsigned long long saturated_samples = 0;
    };

    bool prepareStereoBayer(const GpuBayerFrame& input,
                            parallax::cuda::CudaBuffer& left,
                            parallax::cuda::CudaBuffer& right,
                            std::uint16_t black_level,
                            cudaStream_t stream);

    bool formCanonicalStereo(const parallax::cuda::CudaBuffer& left_linear_rgb16,
                            const parallax::cuda::CudaBuffer& right_linear_rgb16,
                            StereoRgbFrame& rgb_output,
                            StereoGrayFrame& gray_output,
                            const CanonicalRgbParameters& parameters,
                            cudaStream_t stream);

    bool collectIspStatistics(const parallax::cuda::CudaBuffer& linear_rgb16,
                            DeviceIspStatistics* output,
                            float linear_white_level,
                            std::uint32_t sample_stride,
                            cudaStream_t stream);

}
