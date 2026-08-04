#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace parallax::cuda {

    class CudaBuffer {
        public:
            CudaBuffer() = default;

            CudaBuffer(std::uint32_t width, std::uint32_t height, std::uint32_t channels, std::size_t element_size);

            ~CudaBuffer();

            CudaBuffer(const CudaBuffer&) = delete;
            CudaBuffer& operator=(const CudaBuffer&) = delete;

            CudaBuffer(CudaBuffer&& other) noexcept;
            CudaBuffer& operator=(CudaBuffer&& other) noexcept;

            bool allocate(std::uint32_t width, std::uint32_t height, std::uint32_t channels, std::size_t element_size);

            void release() noexcept;

            bool uploadAsync(const void* host_data, std::size_t host_pitch, cudaStream_t stream);

            bool downloadAsync(void* host_data, std::size_t host_pitch, cudaStream_t stream) const;

            bool copyFromAsync(const CudaBuffer& source, cudaStream_t stream);

            [[nodiscard]] bool isAllocated() const noexcept;

            [[nodiscard]] void* data() noexcept;
            [[nodiscard]] const void* data() const noexcept;

            template<typename T>
            [[nodiscard]] T* dataAs() noexcept {
                return static_cast<T*>(device_ptr_);
            }

            template<typename T>
            [[nodiscard]] const T* dataAs() const noexcept {
                return static_cast<const T*>(device_ptr_);
            }

            [[nodiscard]] std::uint32_t width() const noexcept;
            [[nodiscard]] std::uint32_t height() const noexcept;
            [[nodiscard]] std::uint32_t channels() const noexcept;

            [[nodiscard]] std::size_t elementSize() const noexcept;
            [[nodiscard]] std::size_t bytesPerPixel() const noexcept;
            [[nodiscard]] std::size_t rowBytes() const noexcept;
            [[nodiscard]] std::size_t pitch() const noexcept;

            // Logical payload size, excluding CUDA pitch padding.
            [[nodiscard]] std::size_t logicalBytes() const noexcept;

            // Actual allocated size, including pitch padding.
            [[nodiscard]] std::size_t allocatedBytes() const noexcept;

        private:
            void moveFrom(CudaBuffer&& other) noexcept;
            void resetMetadata() noexcept;

            void* device_ptr_ = nullptr;

            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;
            std::uint32_t channels_ = 0;

            std::size_t element_size_ = 0;
            std::size_t pitch_ = 0;
        };

} // namespace parallax::cuda