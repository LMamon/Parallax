#pragma once

#include "px_isp/cuda_buffer.cuh"

#include <nvcv/Tensor.hpp>
#include <cuda_runtime.h>

namespace px_isp {

class CvcudaUtils {
public:
    static nvcv::Tensor createTensor(
        CudaBuffer& buffer,
        cudaStream_t stream);

    static nvcv::Tensor createTensor(
        const CudaBuffer& buffer,
        cudaStream_t stream);
};

} // namespace px_isp