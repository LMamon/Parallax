#pragma once

#include "px_isp/cuda_buffer.hpp"

class BlackLevel {
    public:
        BlackLevel() = default;
        ~BlackLevel() = default;

        bool process(CudaBuffer& left, CudaBuffer& right);
};