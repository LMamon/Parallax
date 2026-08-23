#pragma once

#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>

namespace parallax::core {
    struct SensorFrame {
        const parallax::isp::RectifiedStereoFrame* rgb = nullptr;
        const parallax::isp::StereoMatchFrame* stereo = nullptr;
        const parallax::isp::DepthFrame* depth = nullptr;

        // SensorFrame is still the current per-frame bundle for now.
        // using the same metadata contract here keeps sequence/timestamp semantics
        // consistent while the runtime moves over to individual products.
        ProductMetadata metadata{};  
        parallax::pose::CharucoPoseResult pose;
    };
}