#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>

#include <cstdint>

namespace parallax::core {
    struct SensorFrame {
        const parallax::isp::RectifiedStereoFrame* rgb = nullptr;
        const parallax::isp::StereoMatchFrame* stereo = nullptr;
        const parallax::isp::DepthFrame* depth = nullptr;

        std::uint64_t sequence = 0;
        std::chrono::steady_clock::time_point timestamp;
        parallax::pose::CharucoPoseResult pose;
    };
    
}