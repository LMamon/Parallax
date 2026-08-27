#pragma once

#include <parallax/cuda/cuda_buffer.cuh>

#include <vpi/Image.h>

namespace parallax::vpi {
    class ImageWrapper {
        public:
            ImageWrapper() = default;
            ~ImageWrapper();

            ImageWrapper(const ImageWrapper&) = delete;
            ImageWrapper& operator=(const ImageWrapper&) = delete;

            ImageWrapper(ImageWrapper&& other) noexcept;
            ImageWrapper& operator=(ImageWrapper&& other) noexcept;

            bool create(const parallax::cuda::CudaBuffer& buffer, VPIImageFormat format);
            bool rebind(const parallax::cuda::CudaBuffer& buffer, VPIImageFormat format);

            void release() noexcept;

            [[nodiscard]] VPIImage handle() const noexcept { return image_; }
            [[nodiscard]] bool valid() const noexcept { return image_ != nullptr; }

        private: 
            VPIImage image_ = nullptr;
    };
}