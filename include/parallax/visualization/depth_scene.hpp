#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace parallax::visualization {

    // Visualization geometry only. Full-resolution graph depth is not retained.
    inline std::vector<std::array<float, 3>> buildDepthScenePoints(const float* depth,
                                                                   std::uint32_t width,
                                                                   std::uint32_t height,
                                                                   std::uint32_t sample_stride,
                                                                   const std::array<double, 12>& projection) {

        std::vector<std::array<float, 3>> points;
        if (depth == nullptr || width == 0 || height == 0 || sample_stride == 0) return points;

        const double fx = projection[0];
        const double fy = projection[5];
        const double cx = projection[2];
        const double cy = projection[6];

        if (!std::isfinite(fx) || !std::isfinite(fy) ||
            !std::isfinite(cx) || !std::isfinite(cy) ||
            fx <= 0.0 || fy <= 0.0) return points;

        const std::size_t columns = (width + sample_stride - 1U) / sample_stride;
        const std::size_t rows = (height + sample_stride - 1U) / sample_stride;
        points.reserve(columns * rows);

        for (std::uint32_t v = 0; v < height; v += sample_stride) {
            for (std::uint32_t u = 0; u < width; u += sample_stride) {
                const float z = depth[static_cast<std::size_t>(v) * width + u];
                if (!std::isfinite(z) || z <= 0.0F) continue;

                const float x = static_cast<float>((static_cast<double>(u) - cx) * z / fx);
                const float y = static_cast<float>((static_cast<double>(v) - cy) * z / fy);
                if (!std::isfinite(x) || !std::isfinite(y)) continue;

                points.push_back({x, y, z});
            }
        }
        return points;
    }

}
