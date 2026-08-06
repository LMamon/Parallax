#pragma once

#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/camera/pixel_formats.hpp>

#include <cstdint>

namespace parallax::isp {

    enum class PixelFormat {
        BayerRG10,
        RGB8,
        RGBA8,
        Float32,
        Disparity16,
        Depth32F
    };

    struct GpuBayerFrame {
        parallax::cuda::CudaBuffer buffer;

        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t fourcc = 0;

        parallax::camera::BayerPattern pattern = parallax::camera::BayerPattern::GRBG;
    };

    struct StereoRgbFrame {
        parallax::cuda::CudaBuffer left;
        parallax::cuda::CudaBuffer right;

        std::uint32_t width = 0;
        std::uint32_t height = 0;
        static constexpr std::uint32_t Channels = 3;
        static constexpr PixelFormat Format = PixelFormat::RGB8;
    };

    struct RectifiedStereoFrame {
        parallax::cuda::CudaBuffer left;
        parallax::cuda::CudaBuffer right;

        std::uint32_t width = 0;
        std::uint32_t height = 0;

        static constexpr std::uint32_t Channels = 3;
    };

    struct DisparityFrame {
        parallax::cuda::CudaBuffer image;

        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };
}