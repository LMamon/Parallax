#pragma once

#include <cstdint>
#include <vector>

namespace parallax::lidar {

    /**
     * One valid polar measurement from a LiDAR scan.
     *
     * The SLAMTEC fixed-point representation is normalized at the hardware
     * boundary so downstream Parallax code does not depend on vendor units.
     */
    struct LidarPoint {
        float angle_rad{0.0f};
        float range_m{0.0f};
        std::uint8_t quality{0};
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
            return !points.empty();
        }
    };

}