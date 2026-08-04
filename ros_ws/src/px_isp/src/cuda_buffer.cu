#include "px_isp/cuda_buffer.cuh"

#include <utility>

namespace px_isp {

CudaBuffer::CudaBuffer(std::uint32_t width, std::uint32_t height, std::uint32_t channels, std::size_t element_size) {
    allocate(width, height, channels, element_size);
}

CudaBuffer::~CudaBuffer() {
    release();
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept {
    moveFrom(std::move(other));
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
        release();
        moveFrom(std::move(other));
    }

    return *this;
}

void CudaBuffer::moveFrom(CudaBuffer&& other) noexcept {
    device_ptr_ = other.device_ptr_;

    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;

    element_size_ = other.element_size_;
    pitch_ = other.pitch_;

    other.device_ptr_ = nullptr;
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.element_size_ = 0;
    other.pitch_ = 0;
}

bool CudaBuffer::allocate(std::uint32_t width, std::uint32_t height, std::uint32_t channels, std::size_t element_size) {
    release();

    width_ = width;
    height_ = height;
    channels_ = channels;
    element_size_ = element_size;

    const std::size_t row_bytes = width * channels * element_size;

    cudaError_t err = cudaMallocPitch(&device_ptr_,
                                        &pitch_,
                                        row_bytes,
                                        height);

    if (err != cudaSuccess) {
        release();
        return false;
    }

    return true;
}

void CudaBuffer::release() {
    if (device_ptr_) {
        cudaDeviceSynchronize();
        cudaFree(device_ptr_);
        device_ptr_ = nullptr;
    }

    width_ = 0;
    height_ = 0;
    channels_ = 0;
    element_size_ = 0;
    pitch_ = 0;
}

bool CudaBuffer::isAllocated() const {
    return device_ptr_ != nullptr;
}

void* CudaBuffer::data() {
    return device_ptr_;
}

const void* CudaBuffer::data() const {
    return device_ptr_;
}

std::uint32_t CudaBuffer::width() const {
    return width_;
}

std::uint32_t CudaBuffer::height() const {
    return height_;
}

std::uint32_t CudaBuffer::channels() const {
    return channels_;
}

std::size_t CudaBuffer::elementSize() const {
    return element_size_;
}

std::size_t CudaBuffer::rowBytes() const {
    return width_ * channels_ * element_size_;
}

std::size_t CudaBuffer::pitch() const {
    return pitch_;
}

std::size_t CudaBuffer::allocatedBytes() const {
    return pitch_ * height_;
}

bool CudaBuffer::upload(const void* host_data, std::size_t host_pitch, cudaStream_t stream) {
    if (!device_ptr_ || !host_data) return false;

    cudaError_t err = cudaMemcpy2DAsync(device_ptr_,
                                        pitch_,
                                        host_data,
                                        host_pitch,
                                        rowBytes(),
                                        height_,
                                        cudaMemcpyHostToDevice,
                                        stream);

    return err == cudaSuccess;
}

bool CudaBuffer::download(void* host_data, std::size_t host_pitch, cudaStream_t stream) const {
    if (!device_ptr_ || !host_data) return false;

    cudaError_t err = cudaMemcpy2DAsync(host_data,
                                        host_pitch,
                                        device_ptr_,
                                        pitch_,
                                        rowBytes(),
                                        height_,
                                        cudaMemcpyDeviceToHost,
                                        stream);

    if (err != cudaSuccess) return false;

    return cudaStreamSynchronize(stream) == cudaSuccess;
}

bool CudaBuffer::copyFrom(const CudaBuffer& source, cudaStream_t stream) {
    if (!device_ptr_ || !source.device_ptr_) return false;

    if (width_ != source.width_ || height_ != source.height_ ||
        channels_ != source.channels_ || element_size_ != source.element_size_) {

        return false;
    }

    cudaError_t err = cudaMemcpy2DAsync(device_ptr_,
                                        pitch_,
                                        source.device_ptr_,
                                        source.pitch_,
                                        rowBytes(),
                                        height_,
                                        cudaMemcpyDeviceToDevice,
                                        stream);

    if (err != cudaSuccess) return false;

    return cudaStreamSynchronize(stream) == cudaSuccess;
}
} // namespace px_isp

