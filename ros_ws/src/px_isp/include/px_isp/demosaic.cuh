#pragma once

#include "px_isp/cuda_buffer.cuh"

namespace px_isp {
    class Demosaic {
        public:
            Demosaic() = default;
            ~Demosaic() = default;

            bool process(px_isp::CudaBuffer& left, px_isp::CudaBuffer& right);
    };
}