#include <parallax/perception/efficientvit_sam_cuda.hpp>

#include <cuda_runtime.h>

namespace parallax::perception {

    namespace {

        __global__ void preprocess_kernel(const std::uint8_t* input,
                                          std::size_t input_pitch_bytes,
                                          int input_width,
                                          int input_height,
                                          float* output,
                                          int output_width,
                                          int output_height) {

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= 512 || y >= 512) return;

            constexpr float mean[3] = {123.675F, 116.28F, 103.53F};
            constexpr float std[3] = {58.395F, 57.12F, 57.375F};

            const int plane = 512 * 512;
            const int output_index = y * 512 + x;

            if (x >= output_width || y >= output_height) {
                output[output_index] = 0.0F;
                output[plane + output_index] = 0.0F;
                output[2 * plane + output_index] = 0.0F;
                return;
            }

            const float source_x = (static_cast<float>(x) + 0.5F) *
                                    static_cast<float>(input_width) /
                                    static_cast<float>(output_width) - 0.5F;

            const float source_y = (static_cast<float>(y) + 0.5F) *
                                    static_cast<float>(input_height) /
                                    static_cast<float>(output_height) - 0.5F;

            int x0 = static_cast<int>(floorf(source_x));
            int y0 = static_cast<int>(floorf(source_y));

            const float dx = source_x - static_cast<float>(x0);
            const float dy = source_y - static_cast<float>(y0);

            x0 = max(0, min(x0, input_width - 1));
            y0 = max(0, min(y0, input_height - 1));

            const int x1 = min(x0 + 1, input_width - 1);
            const int y1 = min(y0 + 1, input_height - 1);

            const auto* row0 = input + y0 * input_pitch_bytes;
            const auto* row1 = input + y1 * input_pitch_bytes;

            for (int channel = 0; channel < 3; ++channel) {
                const float p00 = row0[x0 * 3 + channel];
                const float p01 = row0[x1 * 3 + channel];
                const float p10 = row1[x0 * 3 + channel];
                const float p11 = row1[x1 * 3 + channel];

                const float top = p00 + dx * (p01 - p00);
                const float bottom = p10 + dx * (p11 - p10);
                const float value = top + dy * (bottom - top);

                output[channel * plane + output_index] = (value - mean[channel]) / std[channel];
            }
        }

        __global__ void prepare_box_kernel(float x0,
                                           float y0,
                                           float x1,
                                           float y1,
                                           float scale_x,
                                           float scale_y,
                                           float* point_coords,
                                           float* point_labels) {

            if (threadIdx.x != 0 || blockIdx.x != 0) return;

            point_coords[0] = x0 * scale_x;
            point_coords[1] = y0 * scale_y;
            point_coords[2] = x1 * scale_x;
            point_coords[3] = y1 * scale_y;

            point_labels[0] = 2.0F;
            point_labels[1] = 3.0F;
        }


        __device__ float bilinear_sample(const float* image, int width, int height, float x, float y) {
            int x0 = static_cast<int>(floorf(x));
            int y0 = static_cast<int>(floorf(y));

            const float dx = x - static_cast<float>(x0);
            const float dy = y - static_cast<float>(y0);

            x0 = max(0, min(x0, width - 1));
            y0 = max(0, min(y0, height - 1));

            const int x1 = min(x0 + 1, width - 1);
            const int y1 = min(y0 + 1, height - 1);

            const float p00 = image[y0 * width + x0];
            const float p01 = image[y0 * width + x1];
            const float p10 = image[y1 * width + x0];
            const float p11 = image[y1 * width + x1];

            const float top = p00 + dx * (p01 - p00);
            const float bottom = p10 + dx * (p11 - p10);

            return top + dy * (bottom - top);
        }


        __global__ void postprocess_mask_kernel(const float* low_res_mask,
                                                int prompt_width,
                                                int prompt_height,
                                                std::uint8_t* output,
                                                int output_width,
                                                int output_height) {

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= output_width || y >= output_height) return;

            // Reproduce the SAM 1024-space crop/resize geometry without a
            // persistent 1024x1024 intermediate mask.
            const float prompt_x = (static_cast<float>(x) + 0.5F) *
                                    static_cast<float>(prompt_width) /
                                    static_cast<float>(output_width) - 0.5F;

            const float prompt_y = (static_cast<float>(y) + 0.5F) *
                                    static_cast<float>(prompt_height) /
                                    static_cast<float>(output_height) - 0.5F;

            const float low_x = (prompt_x + 0.5F) * 256.0F / 1024.0F - 0.5F;
            const float low_y = (prompt_y + 0.5F) * 256.0F / 1024.0F - 0.5F;
            const float value = bilinear_sample(low_res_mask, 256, 256, low_x, low_y);

            output[y * output_width + x] = value > 0.0F ? 255 : 0;
        }
    }

    bool preprocess_efficientvit_sam(const std::uint8_t* input,
                                     std::size_t input_pitch_bytes,
                                     int input_width,
                                     int input_height,
                                     float* output,
                                     int output_width,
                                     int output_height,
                                     cudaStream_t stream) {

        if (!input || !output || !stream) return false;

        const dim3 block(16, 16);
        const dim3 grid((512 + block.x - 1) / block.x, (512 + block.y - 1) / block.y);

        preprocess_kernel<<<grid, block, 0, stream>>>(input,
                                                      input_pitch_bytes,
                                                      input_width,
                                                      input_height,
                                                      output,
                                                      output_width,
                                                      output_height);

        return cudaGetLastError() == cudaSuccess;
    }

    bool prepare_efficientvit_sam_box(float x0,
                                      float y0,
                                      float x1,
                                      float y1,
                                      int original_width,
                                      int original_height,
                                      int prompt_width,
                                      int prompt_height,
                                      float* point_coords,
                                      float* point_labels,
                                      cudaStream_t stream) {

        if (!point_coords || !point_labels || !stream) return false;

        const float scale_x = static_cast<float>(prompt_width) / static_cast<float>(original_width);
        const float scale_y = static_cast<float>(prompt_height) / static_cast<float>(original_height);

        prepare_box_kernel<<<1, 1, 0, stream>>>(x0,
                                                y0,
                                                x1,
                                                y1,
                                                scale_x,
                                                scale_y,
                                                point_coords,
                                                point_labels);

        return cudaGetLastError() == cudaSuccess;
    }

    bool postprocess_efficientvit_sam_mask(const float* low_res_mask,
                                           int prompt_width,
                                           int prompt_height,
                                           std::uint8_t* output,
                                           int output_width,
                                           int output_height,
                                           cudaStream_t stream) {

        if (!low_res_mask || !output ||
            !stream || prompt_width <= 0 ||
            prompt_height <= 0 || output_width <= 0 || output_height <= 0) {

            return false;
        }

        const dim3 block(16, 16);
        const dim3 grid((output_width + block.x - 1) / block.x, (output_height + block.y - 1) / block.y);

        postprocess_mask_kernel<<<grid, block, 0, stream>>>(low_res_mask,
                                                            prompt_width,
                                                            prompt_height,
                                                            output,
                                                            output_width,
                                                            output_height);

        return cudaGetLastError() == cudaSuccess;
    }
}