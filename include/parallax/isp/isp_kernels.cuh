#pragma once

#include <parallax/cuda/cuda_buffer.cuh>
#include <parallax/isp/frame_types.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace parallax::isp {

bool prepareStereoBayer(const GpuBayerFrame& input,
                        parallax::cuda::CudaBuffer& left,
                        parallax::cuda::CudaBuffer& right,
                        std::uint16_t black_level,
                        cudaStream_t stream);

}
