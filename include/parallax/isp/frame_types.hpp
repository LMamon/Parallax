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
        Depth32F,
        Gray8
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
    
    struct StereoGrayFrame {
        parallax::cuda::CudaBuffer left;
        parallax::cuda::CudaBuffer right;

        std::uint32_t width = 0;
        std::uint32_t height = 0;

        static constexpr std::uint32_t Channels = 1;
        static constexpr PixelFormat Format = PixelFormat::Gray8;
    };

    struct RectifiedStereoFrame {
        parallax::cuda::CudaBuffer left;
        parallax::cuda::CudaBuffer right;

        std::uint32_t width = 0;
        std::uint32_t height = 0;

        static constexpr std::uint32_t Channels = 3;
        static constexpr PixelFormat Format = PixelFormat::RGB8;
    };

    struct RectifiedStereoGrayFrame {
        parallax::cuda::CudaBuffer left;
        parallax::cuda::CudaBuffer right;

        std::uint32_t width = 0;
        std::uint32_t height = 0;

        static constexpr std::uint32_t Channels = 1;
        static constexpr PixelFormat Format = PixelFormat::Gray8;
    };
    
    struct StereoMatchFrame {
        parallax::cuda::CudaBuffer disparity;
        parallax::cuda::CudaBuffer confidence;
        
        std::uint32_t width = 0;
        std::uint32_t height = 0;

        // VPI S16 disparity is Q10.5 fixed point:
        // disparity_pixels = stored_value / 32.0f
        static constexpr float DisparityScale = 32.0f;
    };

    struct DepthFrame {
        parallax::cuda::CudaBuffer depth;

        std::uint32_t width = 0;
        std::uint32_t height = 0;

        static constexpr PixelFormat Format = PixelFormat::Depth32F;
    };
}