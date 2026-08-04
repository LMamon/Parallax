#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace px_isp {
    class CudaBuffer {
        public:
            CudaBuffer() = default;

            CudaBuffer(std::uint32_t width,
                        std::uint32_t height,
                        std::uint32_t channels,
                        std::size_t element_size);

            ~CudaBuffer();

            CudaBuffer(const CudaBuffer&) = delete;
            CudaBuffer& operator=(const CudaBuffer&) = delete;

            CudaBuffer(CudaBuffer&& other) noexcept;
            CudaBuffer& operator=(CudaBuffer&& other) noexcept;

            bool allocate(std::uint32_t width,
                        std::uint32_t height,
                        std::uint32_t channels,
                        std::size_t element_size);

            void release();

            bool upload(const void* host_data, std::size_t host_pitch, cudaStream_t stream = nullptr);
            bool download(void* host_data, std::size_t host_pitch, cudaStream_t stream = nullptr) const;

            bool copyFrom(const CudaBuffer& source, cudaStream_t stream = nullptr);
            bool isAllocated() const;

            void* data();
            const void* data() const;

            template<typename T>
            T* dataAs() {
                return static_cast<T*>(device_ptr_);
            }

            template<typename T>
            const T* dataAs() const {
                return static_cast<const T*>(device_ptr_);
            }

            std::uint32_t width() const;
            std::uint32_t height() const;
            std::uint32_t channels() const;

            std::size_t elementSize() const;
            std::size_t rowBytes() const;
            std::size_t pitch() const;
            std::size_t allocatedBytes() const;

        private:
            void moveFrom(CudaBuffer&& other) noexcept;

            void* device_ptr_ = nullptr;

            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;
            std::uint32_t channels_ = 0;

            std::size_t element_size_ = 0;
            std::size_t pitch_ = 0;
        };

}  // namespace px_isp