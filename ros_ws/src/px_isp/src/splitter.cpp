#include "px_isp/splitter.cuh"

#include <cuda_runtime.h>

namespace px_isp {

bool Splitter::process(const CudaBuffer& input, CudaBuffer& left, CudaBuffer& right, cudaStream_t stream) {
    if (!input.isAllocated() ||
        !left.isAllocated() || !right.isAllocated()) {
        return false;
    }

    const std::size_t half_row_bytes = input.rowBytes() / 2;

    if (left.width() * 2 != input.width() || right.width() * 2 != input.width() ||
        left.height() != input.height() || right.height() != input.height()) {
        return false;
    }
    // Left image
    cudaError_t err = cudaMemcpy2DAsync(left.data(),
                                        left.pitch(),
                                        input.data(),
                                        input.pitch(),
                                        half_row_bytes,
                                        input.height(),
                                        cudaMemcpyDeviceToDevice,
                                        stream);

    if (err != cudaSuccess) return false;

    // Right image
    const auto* right_src = static_cast<const std::uint8_t*>(input.data()) + half_row_bytes;

    err = cudaMemcpy2DAsync(right.data(),
                            right.pitch(),
                            right_src,
                            input.pitch(),
                            half_row_bytes,
                            input.height(),
                            cudaMemcpyDeviceToDevice,
                            stream);

    if (err != cudaSuccess) return false;

    return true;
}

} // namespace px_isp