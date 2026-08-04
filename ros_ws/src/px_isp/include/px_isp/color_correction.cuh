#pragma once

#include "px_isp/cuda_buffer.cuh"

namespace px_isp {
    class ColorCorrection {
        public:
            ColorCorrection() = default;
            ~ColorCorrection() = default;

            bool process(CudaBuffer& left, CudaBuffer& right);
    };
}