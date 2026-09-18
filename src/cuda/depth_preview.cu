#include <parallax/cuda/depth_preview.cuh>
#include <cstddef>
#include <cstdint>
namespace parallax::cuda {
namespace {
__global__ void downsampleDepthNearestKernel(const float* source, std::size_t source_pitch,
                                              float* destination, std::size_t destination_pitch,
                                              std::uint32_t destination_width,
                                              std::uint32_t destination_height,
                                              std::uint32_t stride) {
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= destination_width || y >= destination_height) return;
    const auto* source_row = reinterpret_cast<const float*>(
        reinterpret_cast<const std::uint8_t*>(source) +
        static_cast<std::size_t>(y * stride) * source_pitch);
    auto* destination_row = reinterpret_cast<float*>(
        reinterpret_cast<std::uint8_t*>(destination) +
        static_cast<std::size_t>(y) * destination_pitch);
    destination_row[x] = source_row[x * stride];
}
}
bool downsampleDepthNearest(const CudaBuffer& source, CudaBuffer& destination,
                            std::uint32_t stride, cudaStream_t stream) {
    if (!source.isAllocated() || !destination.isAllocated() || stream == nullptr || stride == 0) return false;
    if (source.channels() != 1 || destination.channels() != 1 ||
        source.elementSize() != sizeof(float) || destination.elementSize() != sizeof(float)) return false;
    const std::uint32_t expected_width = (source.width() + stride - 1U) / stride;
    const std::uint32_t expected_height = (source.height() + stride - 1U) / stride;
    if (destination.width() != expected_width || destination.height() != expected_height) return false;
    const dim3 block(16,16);
    const dim3 grid((destination.width()+block.x-1)/block.x,(destination.height()+block.y-1)/block.y);
    downsampleDepthNearestKernel<<<grid,block,0,stream>>>(source.dataAs<float>(),source.pitch(),
        destination.dataAs<float>(),destination.pitch(),destination.width(),destination.height(),stride);
    return cudaGetLastError() == cudaSuccess;
}
}
