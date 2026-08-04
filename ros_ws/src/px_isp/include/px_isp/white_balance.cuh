#pragma once

#include "px_isp/cuda_buffer.cuh"

namespace px_isp {
    class WhiteBalance {
        public:
            WhiteBalance() = default;
            ~WhiteBalance() = default;

            bool process(CudaBuffer& left, CudaBuffer& right);
    };
}