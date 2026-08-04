#pragma once

#include "px_isp/cuda_buffer.cuh"

namespace px_isp {
    class Gamma {
        public:
            Gamma() = default;
            ~Gamma() = default;

            bool process(CudaBuffer& left, CudaBuffer& right);
    };
}