#pragma once

#include <parallax/isp/frame_types.hpp>

#include <cstdint>

namespace parallax::core {
    struct SensorFrame {
        const parallax::isp::RectifiedStereoFrame* rgb = nullptr;
        const parallax::isp::StereoMatchFrame* stereo = nullptr;

        std::uint64_t sequence = 0;
        std::chrono::steady_clock::time_point timestamp;
    };
    
}