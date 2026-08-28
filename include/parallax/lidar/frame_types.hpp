#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace parallax::lidar {

    /**
     * One polar sample from a completed LiDAR revolution.
     *
     * Samples remain in angular order even when the sensor reports no return.
     * This preserves the scan geometry established by SLAMTEC's
     * ascendScanData(), which reconstructs missing-sample angles using the
     * scan's 360 / sample-count increment.
     *
     * range_m is meaningful only when valid == true.
     */
    struct LidarPoint {
        float angle_rad{0.0f};
        float range_m{0.0f};
        std::uint8_t quality{0};
        bool valid{false};
    };

    /**
     * One completed LiDAR scan.
     *
     * LiDAR remains asynchronous from the camera pipeline. Sensor timing enters
     * ProductMetadata when this payload is exposed through a graph producer.
     */
    struct LidarScan {
        std::vector<LidarPoint> points;

        [[nodiscard]] bool valid() const noexcept {
            return std::any_of(points.begin(), points.end(), [](const LidarPoint& point) {
                    return point.valid;
                });
        }
    };

}