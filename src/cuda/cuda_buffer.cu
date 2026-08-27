#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/core/runtime_metrics.hpp>

#include <limits>
#include <utility>

namespace parallax::cuda {

    namespace {

    bool multiplicationWouldOverflow(std::size_t lhs, std::size_t rhs) noexcept {
        if (lhs == 0 || rhs == 0) return false;

        return lhs > std::numeric_limits<std::size_t>::max() / rhs;
    }

    }

    CudaBuffer::CudaBuffer(std::uint32_t width, std::uint32_t height, std::uint32_t channels, std::size_t element_size) {
        allocate(width, height, channels, element_size);
    }

    CudaBuffer::~CudaBuffer() { release(); }

    CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept { moveFrom(std::move(other)); }

    CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            moveFrom(std::move(other));
        }

        return *this;
    }

    bool CudaBuffer::allocate(std::uint32_t width, std::uint32_t height, std::uint32_t channels, std::size_t element_size) {
        release();

        if (width == 0 || height == 0 || channels == 0 || element_size == 0) {
            return false;
        }

        const std::size_t width_value = static_cast<std::size_t>(width);
        const std::size_t channels_value = static_cast<std::size_t>(channels);

        if (multiplicationWouldOverflow(width_value, channels_value)) return false;
        const std::size_t pixel_count_per_row = width_value * channels_value;

        if (multiplicationWouldOverflow(pixel_count_per_row, element_size)) return false;
        const std::size_t row_bytes = pixel_count_per_row * element_size;

        void* allocation = nullptr;
        std::size_t allocation_pitch = 0;

        const cudaError_t error = cudaMallocPitch(&allocation, 
                                                  &allocation_pitch, 
                                                  row_bytes, 
                                                  static_cast<std::size_t>(height));

        if (error != cudaSuccess) return false;

        device_ptr_ = allocation;

        width_ = width;
        height_ = height;
        channels_ = channels;

        element_size_ = element_size;
        pitch_ = allocation_pitch;

        auto& metrics = parallax::core::runtime_metrics();
        ++metrics.cuda_allocations;
        metrics.cuda_allocated_bytes += allocatedBytes();

        return true;
    }

    void CudaBuffer::release() noexcept {
        if (device_ptr_ != nullptr) {
            /*
             * cudaFree may synchronize work that uses this allocation.
             * deliberately avoided cudaDeviceSynchronize(), which would
             * stall unrelated work on the entire device.
             *
             * The owning pipeline must ensure that no future operation
             * references this buffer before destruction.
             */
            cudaFree(device_ptr_);
            ++parallax::core::runtime_metrics().cuda_frees;
        }
        resetMetadata();
    }

    bool CudaBuffer::uploadAsync(const void* host_data, std::size_t host_pitch, cudaStream_t stream) {
        if (!isAllocated() || host_data == nullptr) return false;

        if (host_pitch < rowBytes()) return false;

        const cudaError_t error = cudaMemcpy2DAsync(device_ptr_,
                                                    pitch_,
                                                    host_data,
                                                    host_pitch,
                                                    rowBytes(),
                                                    static_cast<std::size_t>(height_),
                                                    cudaMemcpyHostToDevice,
                                                    stream);

        if (error != cudaSuccess) return false;

        auto& metrics = parallax::core::runtime_metrics();
        ++metrics.host_to_device_transfers;
        metrics.host_to_device_bytes += logicalBytes();

        return true;
    }

    bool CudaBuffer::downloadAsync(void* host_data, std::size_t host_pitch, cudaStream_t stream) const {
        if (!isAllocated() || host_data == nullptr) return false;

        if (host_pitch < rowBytes()) return false;

        const cudaError_t error = cudaMemcpy2DAsync(host_data,
                                                    host_pitch,
                                                    device_ptr_,
                                                    pitch_,
                                                    rowBytes(),
                                                    static_cast<std::size_t>(height_),
                                                    cudaMemcpyDeviceToHost,
                                                    stream);

        if (error != cudaSuccess) return false;

        auto& metrics = parallax::core::runtime_metrics();
        ++metrics.device_to_host_transfers;
        metrics.device_to_host_bytes += logicalBytes();

        return true;
    }

    bool CudaBuffer::copyFromAsync(const CudaBuffer& source, cudaStream_t stream) {
        if (!isAllocated() || !source.isAllocated()) return false;

        if (width_ != source.width_ || height_ != source.height_ ||
            channels_ != source.channels_ || element_size_ != source.element_size_) {
            
                return false;
        }

        const cudaError_t error = cudaMemcpy2DAsync(device_ptr_,
                                                    pitch_,
                                                    source.device_ptr_,
                                                    source.pitch_,
                                                    rowBytes(),
                                                    static_cast<std::size_t>(height_),
                                                    cudaMemcpyDeviceToDevice,
                                                    stream);

        if (error != cudaSuccess) return false;

        auto& metrics = parallax::core::runtime_metrics();
        ++metrics.device_to_device_transfers;
        metrics.device_to_device_bytes += logicalBytes();

        return true;
    }

    bool CudaBuffer::isAllocated() const noexcept { return device_ptr_ != nullptr; }

    void* CudaBuffer::data() noexcept { return device_ptr_; }
    const void* CudaBuffer::data() const noexcept { return device_ptr_; }

    std::uint32_t CudaBuffer::width() const noexcept { return width_; }
    std::uint32_t CudaBuffer::height() const noexcept { return height_; }

    std::uint32_t CudaBuffer::channels() const noexcept { return channels_; }
    std::size_t CudaBuffer::elementSize() const noexcept { return element_size_; }

    std::size_t CudaBuffer::bytesPerPixel() const noexcept {
        return static_cast<std::size_t>(channels_) * element_size_;
    }

    std::size_t CudaBuffer::rowBytes() const noexcept {
        return static_cast<std::size_t>(width_) * bytesPerPixel();
    }

    std::size_t CudaBuffer::pitch() const noexcept { return pitch_; }

    std::size_t CudaBuffer::logicalBytes() const noexcept {
        return rowBytes() * static_cast<std::size_t>(height_);
    }

    std::size_t CudaBuffer::allocatedBytes() const noexcept {
        return pitch_ * static_cast<std::size_t>(height_);
    }

    void CudaBuffer::moveFrom(CudaBuffer&& other) noexcept {
        device_ptr_ = other.device_ptr_;

        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;

        element_size_ = other.element_size_;
        pitch_ = other.pitch_;

        other.resetMetadata();
    }

    void CudaBuffer::resetMetadata() noexcept {
        device_ptr_ = nullptr;

        width_ = 0;
        height_ = 0;
        channels_ = 0;

        element_size_ = 0;
        pitch_ = 0;
    }
}