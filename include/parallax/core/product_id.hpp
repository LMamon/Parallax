#pragma once

#include <cstdint>

namespace parallax::core {
    // Stable identifiers for products that may be requested, computed, cached,
    // or published by the runtime dependency graph.
    //
    // ProductId describes what data exists, not how it is computed. Algorithms
    // such as VPI rectification, stereo disparity, pose estimation, or detection
    // remain implementation details of the producers that provide these products.
    //
    // using explicit vocab rather than deriving IDs from C++ types.
    // Multiple products may eventually share the same payload type while having
    // different semantic meaning in the graph.

    enum class ProductId : std::uint8_t {
        RawStereo, 
        RgbLeft,
        GrayStereo,
        RectifiedRgb,
        RectifiedGray,
        Disparity,
        Confidence,
        Depth,
        Pose, 
        Projection,
        MarkerDepth,
        Detection,
        Segmentation,
        Track2D,
        Track3D,
        LidarScan
    };
}