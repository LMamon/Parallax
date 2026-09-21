#include <parallax/isp/isp_kernels.cuh>

#include <cmath>
#include <cstdint>

namespace parallax::isp {
    namespace {

        constexpr std::uint16_t Bayer10Maximum = 1023;

        __device__ __forceinline__ float clamp01(float value) { return fminf(1.0F, fmaxf(0.0F, value)); }

        __global__ void prepareStereoBayerKernel(const std::uint16_t* input,
                                                std::size_t input_pitch,
                                                std::uint16_t* left,
                                                std::size_t left_pitch,
                                                std::uint16_t* right,
                                                std::size_t right_pitch,
                                                int combined_width,
                                                int height,
                                                std::uint16_t black_level) {

            const int combined_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            if (combined_x >= combined_width || y >= height) return;

            const int eye_width = combined_width / 2;
            const bool right_eye = combined_x >= eye_width;
            const int local_x = right_eye ? combined_x - eye_width : combined_x;

            const auto* source_row = reinterpret_cast<const std::uint16_t*>(
                reinterpret_cast<const std::uint8_t*>(input) + static_cast<std::size_t>(y) * input_pitch);

            // The Arducam BA10 path stores one left-aligned 10-bit sample per 16-bit word.
            const std::uint16_t raw10 = static_cast<std::uint16_t>((source_row[combined_x] >> 6U) & Bayer10Maximum);
            const std::uint16_t normalized = raw10 > black_level ? static_cast<std::uint16_t>(raw10 - black_level) : 0U;

            auto* destination = right_eye ? right : left;
            const std::size_t destination_pitch = right_eye ? right_pitch : left_pitch;
            auto* destination_row = reinterpret_cast<std::uint16_t*>(reinterpret_cast<std::uint8_t*>(destination) + static_cast<std::size_t>(y) * destination_pitch);

            destination_row[local_x] = normalized;
        }

        __global__ void canonicalRgbKernel(const std::uint16_t* input,
                                        std::size_t input_pitch,
                                        std::uint8_t* rgb,
                                        std::size_t rgb_pitch,
                                        std::uint8_t* gray,
                                        std::size_t gray_pitch,
                                        int width,
                                        int height,
                                        CanonicalRgbParameters parameters) {

            const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            if (x >= width || y >= height) return;

            const auto* source_row = reinterpret_cast<const std::uint16_t*>(reinterpret_cast<const std::uint8_t*>(input) + static_cast<std::size_t>(y) * input_pitch);

            const std::size_t source_offset = static_cast<std::size_t>(x) * 3U;
            const float scale = 1.0F / fmaxf(parameters.linear_white_level, 1.0F);

            const float linear_r = static_cast<float>(source_row[source_offset + 0]) * scale;
            const float linear_g = static_cast<float>(source_row[source_offset + 1]) * scale;
            const float linear_b = static_cast<float>(source_row[source_offset + 2]) * scale;

            // Geometry gets a controlled linear luminance representation, before WB/CCM/gamma.
            const float linear_luma = clamp01(0.299F * linear_r + 0.587F * linear_g + 0.114F * linear_b);
            auto* gray_row = gray + static_cast<std::size_t>(y) * gray_pitch;
            gray_row[x] = static_cast<std::uint8_t>(linear_luma * 255.0F + 0.5F);

            const float wb_r = linear_r * parameters.white_balance[0];
            const float wb_g = linear_g * parameters.white_balance[1];
            const float wb_b = linear_b * parameters.white_balance[2];

            float r = parameters.color_matrix[0] * wb_r + parameters.color_matrix[1] * wb_g + parameters.color_matrix[2] * wb_b;
            float g = parameters.color_matrix[3] * wb_r + parameters.color_matrix[4] * wb_g + parameters.color_matrix[5] * wb_b;
            float b = parameters.color_matrix[6] * wb_r + parameters.color_matrix[7] * wb_g + parameters.color_matrix[8] * wb_b;

            r = powf(clamp01(r), parameters.inverse_gamma);
            g = powf(clamp01(g), parameters.inverse_gamma);
            b = powf(clamp01(b), parameters.inverse_gamma);

            auto* rgb_row = rgb + static_cast<std::size_t>(y) * rgb_pitch;
            const std::size_t destination_offset = static_cast<std::size_t>(x) * 3U;

            rgb_row[destination_offset + 0] = static_cast<std::uint8_t>(r * 255.0F + 0.5F);
            rgb_row[destination_offset + 1] = static_cast<std::uint8_t>(g * 255.0F + 0.5F);
            rgb_row[destination_offset + 2] = static_cast<std::uint8_t>(b * 255.0F + 0.5F);
        }

        __global__ void statisticsKernel(const std::uint16_t* input,
                                        std::size_t input_pitch,
                                        int width,
                                        int height,
                                        float linear_white_level,
                                        std::uint32_t sample_stride,
                                        DeviceIspStatistics* output) {

            const int sample_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int sample_y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

            const int x = sample_x * static_cast<int>(sample_stride);
            const int y = sample_y * static_cast<int>(sample_stride);
            
            if (x >= width || y >= height) return;

            const auto* row = reinterpret_cast<const std::uint16_t*>(reinterpret_cast<const std::uint8_t*>(input) + static_cast<std::size_t>(y) * input_pitch);
            const std::size_t offset = static_cast<std::size_t>(x) * 3U;

            const float scale = 1.0F / fmaxf(linear_white_level, 1.0F);

            const float r = static_cast<float>(row[offset + 0]) * scale;
            const float g = static_cast<float>(row[offset + 1]) * scale;
            const float b = static_cast<float>(row[offset + 2]) * scale;
            const float luma = clamp01(0.299F * r + 0.587F * g + 0.114F * b);
            
            const std::uint32_t bin = min(255U, static_cast<std::uint32_t>(luma * 255.0F));

            atomicAdd(&output->luminance_histogram[bin], 1U);
            atomicAdd(&output->total_samples, 1ULL);

            const float maximum = fmaxf(r, fmaxf(g, b));
            if (maximum >= 0.98F) atomicAdd(&output->saturated_samples, 1ULL);

            // AWB uses midtones only so one clipped lamp or deep shadow does not dominate it.
            if (luma >= 0.125F && luma <= 0.875F && maximum < 0.98F) {
                constexpr float SumScale = 4096.0F;
                
                atomicAdd(&output->red_sum, static_cast<unsigned long long>(fmaxf(r, 0.0F) * SumScale));
                atomicAdd(&output->green_sum, static_cast<unsigned long long>(fmaxf(g, 0.0F) * SumScale));
                atomicAdd(&output->blue_sum, static_cast<unsigned long long>(fmaxf(b, 0.0F) * SumScale));
                atomicAdd(&output->color_samples, 1ULL);
            }
        }

        bool validLinearRgb(const parallax::cuda::CudaBuffer& buffer) {
            return buffer.isAllocated() && buffer.channels() == 3 && buffer.elementSize() == sizeof(std::uint16_t);
        }

        bool launchCanonical(const parallax::cuda::CudaBuffer& input,
                            parallax::cuda::CudaBuffer& rgb,
                            parallax::cuda::CudaBuffer& gray,
                            const CanonicalRgbParameters& parameters,
                            cudaStream_t stream) {

            if (!validLinearRgb(input) || !rgb.isAllocated() || !gray.isAllocated()) return false;
            if (rgb.channels() != 3 || rgb.elementSize() != sizeof(std::uint8_t) ||
                gray.channels() != 1 || gray.elementSize() != sizeof(std::uint8_t)) return false;
            if (input.width() != rgb.width() || input.height() != rgb.height() ||
                input.width() != gray.width() || input.height() != gray.height()) return false;

            constexpr dim3 block(16, 16);
            const dim3 grid((input.width() + block.x - 1U) / block.x,
                            (input.height() + block.y - 1U) / block.y);

            canonicalRgbKernel<<<grid, block, 0, stream>>>(input.dataAs<std::uint16_t>(), input.pitch(),
                                                          rgb.dataAs<std::uint8_t>(), rgb.pitch(),
                                                          gray.dataAs<std::uint8_t>(), gray.pitch(),
                                                          static_cast<int>(input.width()), static_cast<int>(input.height()), parameters);

            return cudaPeekAtLastError() == cudaSuccess;
        }

    }

    bool prepareStereoBayer(const GpuBayerFrame& input,
                            parallax::cuda::CudaBuffer& left,
                            parallax::cuda::CudaBuffer& right,
                            std::uint16_t black_level,
                            cudaStream_t stream) {

        if (!input.buffer.isAllocated() || input.width == 0 || input.height == 0 ||
            (input.width % 2U) != 0U || black_level >= Bayer10Maximum) return false;

        const std::uint32_t eye_width = input.width / 2U;

        if (!left.isAllocated() || !right.isAllocated() ||
            left.width() != eye_width || right.width() != eye_width ||
            left.height() != input.height || right.height() != input.height ||
            left.channels() != 1 || right.channels() != 1 ||
            left.elementSize() != sizeof(std::uint16_t) || right.elementSize() != sizeof(std::uint16_t)) return false;

        constexpr dim3 block(16, 16);
        const dim3 grid((input.width + block.x - 1U) / block.x,
                        (input.height + block.y - 1U) / block.y);

        prepareStereoBayerKernel<<<grid, block, 0, stream>>>(input.buffer.dataAs<std::uint16_t>(), 
                                                                input.buffer.pitch(),
                                                                left.dataAs<std::uint16_t>(), left.pitch(),
                                                                right.dataAs<std::uint16_t>(), right.pitch(),
                                                                static_cast<int>(input.width), 
                                                                static_cast<int>(input.height), 
                                                                black_level);

        return cudaPeekAtLastError() == cudaSuccess;
    }

    bool formCanonicalStereo(const parallax::cuda::CudaBuffer& left_linear_rgb16,
                            const parallax::cuda::CudaBuffer& right_linear_rgb16,
                            StereoRgbFrame& rgb_output,
                            StereoGrayFrame& gray_output,
                            const CanonicalRgbParameters& parameters,
                            cudaStream_t stream) {

        if (!launchCanonical(left_linear_rgb16, rgb_output.left, gray_output.left, parameters, stream)) return false;

        return launchCanonical(right_linear_rgb16, rgb_output.right, gray_output.right, parameters, stream);
    }

    bool collectIspStatistics(const parallax::cuda::CudaBuffer& linear_rgb16,
                            DeviceIspStatistics* output,
                            float linear_white_level,
                            std::uint32_t sample_stride,
                            cudaStream_t stream) {

        if (!validLinearRgb(linear_rgb16) || output == nullptr || linear_white_level <= 0.0F || sample_stride == 0) return false;

        const std::uint32_t sample_width = (linear_rgb16.width() + sample_stride - 1U) / sample_stride;
        const std::uint32_t sample_height = (linear_rgb16.height() + sample_stride - 1U) / sample_stride;

        constexpr dim3 block(16, 16);
        
        const dim3 grid((sample_width + block.x - 1U) / block.x,
                        (sample_height + block.y - 1U) / block.y);

        statisticsKernel<<<grid, block, 0, stream>>>(linear_rgb16.dataAs<std::uint16_t>(), 
                                                        linear_rgb16.pitch(),
                                                        static_cast<int>(linear_rgb16.width()), static_cast<int>(linear_rgb16.height()),
                                                        linear_white_level, 
                                                        sample_stride, 
                                                        output);

        return cudaPeekAtLastError() == cudaSuccess;
    }
}
