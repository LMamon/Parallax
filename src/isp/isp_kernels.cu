#include <parallax/isp/isp_kernels.cuh>

#include <cmath>
#include <cstdint>

namespace parallax::isp {
    namespace {

        constexpr std::uint16_t Bayer10Maximum = 1023;

        __global__ void prepareStereoBayerKernel(const std::uint16_t* input,
                                                std::size_t input_pitch,
                                                std::uint16_t* left,
                                                std::size_t left_pitch,
                                                std::uint16_t* right,
                                                std::size_t right_pitch,
                                                int combined_width,
                                                int height,
                                                std::uint16_t black_level) {
            const int combined_x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
            if (combined_x >= combined_width || y >= height) return;

            const int eye_width = combined_width / 2;
            const bool right_eye = combined_x >= eye_width;
            const int local_x = right_eye ? combined_x - eye_width : combined_x;

            const auto* source_row = reinterpret_cast<const std::uint16_t*>(
                reinterpret_cast<const std::uint8_t*>(input) + static_cast<std::size_t>(y) * input_pitch);

            // The Arducam BA10 path stores one left-aligned 10-bit sample per 16-bit word.
            const std::uint16_t raw10 = static_cast<std::uint16_t>((source_row[combined_x] >> 6U) & Bayer10Maximum);
            const std::uint16_t normalized = raw10 > black_level
                ? static_cast<std::uint16_t>(raw10 - black_level)
                : 0U;

            auto* destination = right_eye ? right : left;
            const std::size_t destination_pitch = right_eye ? right_pitch : left_pitch;
            auto* destination_row = reinterpret_cast<std::uint16_t*>(
                reinterpret_cast<std::uint8_t*>(destination) + static_cast<std::size_t>(y) * destination_pitch);

            destination_row[local_x] = normalized;
        }

        bool prepareStereoBayer(const GpuBayerFrame& input,
                                parallax::cuda::CudaBuffer& left,
                                parallax::cuda::CudaBuffer& right,
                                std::uint16_t black_level,
                                cudaStream_t stream) {
            if (!input.buffer.isAllocated() || input.width == 0 || input.height == 0 ||
                (input.width % 2U) != 0U || black_level >= Bayer10Maximum) return false;

            const std::uint32_t eye_width = input.width / 2U;
            if (!left.isAllocated() || !right.isAllocated() ||
                left.width() != eye_width || right.width() != eye_width ||
                left.height() != input.height || right.height() != input.height ||
                left.channels() != 1 || right.channels() != 1 ||
                left.elementSize() != sizeof(std::uint16_t) || right.elementSize() != sizeof(std::uint16_t)) return false;

            constexpr dim3 block(16, 16);
            const dim3 grid((input.width + block.x - 1U) / block.x,
                            (input.height + block.y - 1U) / block.y);

            prepareStereoBayerKernel<<<grid, block, 0, stream>>>(
                input.buffer.dataAs<std::uint16_t>(), input.buffer.pitch(),
                left.dataAs<std::uint16_t>(), left.pitch(),
                right.dataAs<std::uint16_t>(), right.pitch(),
                static_cast<int>(input.width), static_cast<int>(input.height), black_level);

            return cudaPeekAtLastError() == cudaSuccess;
        }

    }
}