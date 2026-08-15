#include <parallax/isp/demosaic.cuh>
#include <parallax/camera/pixel_formats.hpp>

#include <cstddef>
#include <cstdint>

#include <linux/videodev2.h>
namespace parallax::isp {

namespace {
    constexpr std::uint16_t Bayer10Maximum = 1023;

    __device__ __forceinline__ int clampCoordinate(int value, int minimum, int maximum) {
        return max(minimum, min(value, maximum));
    }

    __device__ __forceinline__ std::uint16_t readBayerPixel(const std::uint16_t* input,
                                                            std::size_t input_pitch,
                                                            int local_x,
                                                            int y,
                                                            int camera_offset_x,
                                                            int camera_width,
                                                            int image_height) {
        /*
        * Clamp within one camera image Interpolation for the left camera
        *  must never sample the right camera and vice versa
        */
        const int clamped_x = clampCoordinate(local_x, 0, camera_width - 1);
        const int clamped_y = clampCoordinate(y, 0, image_height - 1);
        const int source_x = camera_offset_x + clamped_x;

        const auto* row = reinterpret_cast<const std::uint16_t*>(
            reinterpret_cast<const std::uint8_t*>(input) + static_cast<std::size_t>(clamped_y) * input_pitch);

       /*
        * The driver supplies each 10-bit Bayer sample left-aligned in a
        * 16-bit word. Shift right by six to recover the original
        * [0, 1023] sensor value
        */
        return static_cast<std::uint16_t>((row[source_x] >> 6U) & Bayer10Maximum);
    }

    __device__ __forceinline__ std::uint8_t bayer10ToRgb8(std::uint32_t value) {
        value = min(value, static_cast<std::uint32_t>(Bayer10Maximum));

        //Integer conversion from [0, 1023] to [0, 255], with rounding
        return static_cast<std::uint8_t>((value * 255U + 511U) / Bayer10Maximum);
    }

    __device__ __forceinline__ void writeRgbPixel(std::uint8_t* output,
                                                std::size_t output_pitch,
                                                int x,
                                                int y,
                                                std::uint8_t red,
                                                std::uint8_t green,
                                                std::uint8_t blue) {

        auto* row = output + static_cast<std::size_t>(y) * output_pitch;
        const std::size_t pixel_offset = static_cast<std::size_t>(x) * StereoRgbFrame::Channels;

        row[pixel_offset + 0] = red;
        row[pixel_offset + 1] = green;
        row[pixel_offset + 2] = blue;
    }

    __device__ __forceinline__ void writeGrayPixel(std::uint8_t* output,
                                                    std::size_t output_pitch,
                                                    int x,
                                                    int y,
                                                    std::uint8_t red,
                                                    std::uint8_t green,
                                                    std::uint8_t blue) {

        auto* row = output + static_cast<std::size_t>(y) * output_pitch;

        // Integer approximation of:
        // Y = 0.299 R + 0.587 G + 0.114 B
        // Coeffs sum to 256
        const std::uint32_t luminance = 77U  * static_cast<std::uint32_t>(red) +
                                        150U * static_cast<std::uint32_t>(green) +
                                        29U  * static_cast<std::uint32_t>(blue);

        row[x] = static_cast<std::uint8_t>((luminance + 128U) >> 8U);
    }

    __device__ __forceinline__ void demosaicGrbgPixel(const std::uint16_t* input,
                                                        std::size_t input_pitch,
                                                        int local_x,
                                                        int y,
                                                        int camera_offset_x,
                                                        int camera_width,
                                                        int image_height,
                                                        std::uint8_t& red,
                                                        std::uint8_t& green,
                                                        std::uint8_t& blue) {

        const auto sample = [&](int x, int sample_y) {
            return static_cast<std::uint32_t>(readBayerPixel(input,
                                                            input_pitch,
                                                            x,
                                                            sample_y,
                                                            camera_offset_x,
                                                            camera_width,
                                                            image_height));
        };

        const bool even_x = (local_x & 1) == 0;
        const bool even_y = (y & 1) == 0;
        const std::uint32_t center = sample(local_x, y);

        std::uint32_t red_value = 0;
        std::uint32_t green_value = 0;
        std::uint32_t blue_value = 0;

        /*
        * GRBG:
        * G R G R
        * B G B G
        */
        if (even_y && even_x) {
            // Green pixel on a red row.
            green_value = center;
            red_value = (sample(local_x - 1, y) +
                        sample(local_x + 1, y) +
                        1U) / 2U;

            blue_value = (sample(local_x, y - 1) +
                        sample(local_x, y + 1) +
                        1U) / 2U;
        }
        else if (even_y && !even_x) {
            // Red pixel.
            red_value = center;
            green_value = (sample(local_x - 1, y) +
                        sample(local_x + 1, y) +
                        sample(local_x, y - 1) +
                        sample(local_x, y + 1) +
                        2U) / 4U;

            blue_value = (sample(local_x - 1, y - 1) +
                        sample(local_x + 1, y - 1) +
                        sample(local_x - 1, y + 1) +
                        sample(local_x + 1, y + 1) +
                        2U) / 4U;
        }
        else if (!even_y && even_x) {
            // Blue pixel.
            blue_value = center;
            green_value = (sample(local_x - 1, y) +
                        sample(local_x + 1, y) +
                        sample(local_x, y - 1) +
                        sample(local_x, y + 1) +
                        2U) / 4U;

            red_value = (sample(local_x - 1, y - 1) +
                        sample(local_x + 1, y - 1) +
                        sample(local_x - 1, y + 1) +
                        sample(local_x + 1, y + 1) +
                        2U) / 4U;
        }
        else {
            // Green pixel on a blue row.
            green_value = center;
            red_value = (sample(local_x, y - 1) +
                        sample(local_x, y + 1) +
                        1U) / 2U;

            blue_value = (sample(local_x - 1, y) +
                        sample(local_x + 1, y) +
                        1U) / 2U;
        }

        red = bayer10ToRgb8(red_value);
        green = bayer10ToRgb8(green_value);
        blue = bayer10ToRgb8(blue_value);
    }

    __global__ void demosaicAndSplitKernel(const std::uint16_t* input,
                                            std::size_t input_pitch,
                                            std::uint8_t* left_rgb,
                                            std::size_t left_rgb_pitch,
                                            std::uint8_t* right_rgb,
                                            std::size_t right_rgb_pitch,
                                            std::uint8_t* left_gray,
                                            std::size_t left_gray_pitch,
                                            std::uint8_t* right_gray,
                                            std::size_t right_gray_pitch,

                                            int combined_width,
                                            int image_height) {

        const int combined_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

        if (combined_x >= combined_width || y >= image_height) return;

        const int camera_width = combined_width / 2;
        const bool is_right_camera = combined_x >= camera_width;

        const int camera_offset_x = is_right_camera ? camera_width : 0;
        const int local_x = is_right_camera ? combined_x - camera_width : combined_x;

        std::uint8_t red = 0;
        std::uint8_t green = 0;
        std::uint8_t blue = 0;

        /*
        * Bayer parity uses local_x rather than combined_x.
        * Each sensor begins its own GRBG mosaic at local coordinate (0, 0).
        */
        demosaicGrbgPixel(input,
                        input_pitch,
                        local_x,
                        y,
                        camera_offset_x,
                        camera_width,
                        image_height,
                        red,
                        green,
                        blue);
        
        std::uint8_t* rgb_destination = is_right_camera ? right_rgb : left_rgb;
        const std::size_t rgb_pitch = is_right_camera ? right_rgb_pitch : left_rgb_pitch;

        std::uint8_t* gray_destination = is_right_camera ? right_gray : left_gray;
        const std::size_t gray_pitch = is_right_camera ? right_gray_pitch : left_gray_pitch;

        writeRgbPixel(rgb_destination, rgb_pitch, local_x, y, red, green, blue);
        writeGrayPixel(gray_destination, gray_pitch, local_x, y, red, green, blue);
    }

    bool dimensionsAreValid(const GpuBayerFrame& input, const StereoRgbFrame& rgb, const StereoGrayFrame& gray) {
        if (input.width == 0 || input.height == 0) return false;
        if ((input.width % 2U) != 0U) return false;

        const std::uint32_t camera_width = input.width / 2U;

        if (rgb.width != camera_width ||
            rgb.height != input.height ||
            gray.width != camera_width ||
            gray.height != input.height) {

            return false;
        }

        if (rgb.left.width() != camera_width ||
            rgb.right.width() != camera_width ||
            rgb.left.height() != input.height ||
            rgb.right.height() != input.height) {

            return false;
        }

        if (gray.left.width() != camera_width ||
            gray.right.width() != camera_width ||
            gray.left.height() != input.height ||
            gray.right.height() != input.height) {

            return false;
        }

        if (rgb.left.channels() != StereoRgbFrame::Channels ||
            rgb.right.channels() != StereoRgbFrame::Channels ||
            rgb.left.elementSize() != sizeof(std::uint8_t) ||
            rgb.right.elementSize() != sizeof(std::uint8_t)) {

            return false;
        }

        if (gray.left.channels() != StereoGrayFrame::Channels ||
            gray.right.channels() != StereoGrayFrame::Channels ||
            gray.left.elementSize() != sizeof(std::uint8_t) ||
            gray.right.elementSize() != sizeof(std::uint8_t)) {

            return false;
        }
        return true;
    }
}

bool demosaicAndSplit(const GpuBayerFrame& input, StereoRgbFrame& rgb_output, StereoGrayFrame& gray_output, cudaStream_t stream) {
    if (input.pattern != parallax::camera::BayerPattern::GRBG) return false;
    if (!input.buffer.isAllocated() ||
        !rgb_output.left.isAllocated() ||
        !rgb_output.right.isAllocated() ||
        !gray_output.left.isAllocated() ||
        !gray_output.right.isAllocated()) {

        return false;
    }

    if (input.buffer.width() != input.width ||
        input.buffer.height() != input.height ||
        input.buffer.channels() != 1 ||
        input.buffer.elementSize() != sizeof(std::uint16_t)) {

        return false;
    }
    
    if (!dimensionsAreValid(input, rgb_output, gray_output)) return false;

    constexpr dim3 block(16, 16);
    const dim3 grid((input.width + block.x - 1U) / block.x,
                    (input.height + block.y - 1U) / block.y);

    demosaicAndSplitKernel<<<grid, block, 0, stream>>>(input.buffer.dataAs<std::uint16_t>(),
                                                    input.buffer.pitch(),
                                                    rgb_output.left.dataAs<std::uint8_t>(),
                                                    rgb_output.left.pitch(),
                                                    rgb_output.right.dataAs<std::uint8_t>(),
                                                    rgb_output.right.pitch(),
                                                    gray_output.left.dataAs<std::uint8_t>(),
                                                    gray_output.left.pitch(),
                                                    gray_output.right.dataAs<std::uint8_t>(),
                                                    gray_output.right.pitch(),
                                                    static_cast<int>(input.width),
                                                    static_cast<int>(input.height));

    /*
    * check whether the kernel launch was accepted. It deliberately
    * does not synchronize the stream + execution remains asynchronous.
    */
    return cudaPeekAtLastError() == cudaSuccess;
}
}