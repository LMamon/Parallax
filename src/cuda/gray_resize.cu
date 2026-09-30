#include <parallax/cuda/gray_resize.cuh>

#include <cstdint>

namespace parallax::cuda {
namespace {

__global__ void downsampleGray2xKernel(const std::uint8_t* input,
                                       std::size_t input_pitch,
                                       std::uint8_t* output,
                                       std::size_t output_pitch,
                                       std::uint32_t width,
                                       std::uint32_t height) {
    const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const std::uint32_t sx = x * 2U;
    const std::uint32_t sy = y * 2U;
    const auto* row0 = input + static_cast<std::size_t>(sy) * input_pitch;
    const auto* row1 = input + static_cast<std::size_t>(sy + 1U) * input_pitch;
    auto* dst = output + static_cast<std::size_t>(y) * output_pitch;

    const unsigned sum =
        static_cast<unsigned>(row0[sx]) +
        static_cast<unsigned>(row0[sx + 1U]) +
        static_cast<unsigned>(row1[sx]) +
        static_cast<unsigned>(row1[sx + 1U]);
    dst[x] = static_cast<std::uint8_t>((sum + 2U) / 4U);
}

}  // namespace

bool downsampleGray2x(const CudaBuffer& input,
                      CudaBuffer& output,
                      cudaStream_t stream) {
    if (!input.isAllocated() || !output.isAllocated() || stream == nullptr ||
        input.channels() != 1 || output.channels() != 1 ||
        input.elementSize() != sizeof(std::uint8_t) ||
        output.elementSize() != sizeof(std::uint8_t) ||
        input.width() != output.width() * 2U ||
        input.height() != output.height() * 2U) {
        return false;
    }

    const dim3 block(16, 16);
    const dim3 grid((output.width() + block.x - 1U) / block.x,
                    (output.height() + block.y - 1U) / block.y);
    downsampleGray2xKernel<<<grid, block, 0, stream>>>(
        input.dataAs<std::uint8_t>(), input.pitch(),
        output.dataAs<std::uint8_t>(), output.pitch(),
        output.width(), output.height());
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace parallax::cuda
