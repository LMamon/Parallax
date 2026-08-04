#pragma once

#include "px_isp/cuda_buffer.cuh"

namespace px_isp {
    class Splitter {
        public:
            Splitter() = default;
            ~Splitter() = default;

            bool process(const CudaBuffer& input, CudaBuffer& left, CudaBuffer& right, cudaStream_t stream);
    };
}