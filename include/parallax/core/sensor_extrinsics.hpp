#pragma once

#include <array>
#include <string>

namespace parallax::core {

    struct RigidTransformConfig {
        std::string parent_frame;
        std::string child_frame;

        std::array<double, 3> translation_m{};
        std::array<double, 4> rotation_xyzw{};
    };

    class SensorExtrinsics {
        public:
            bool loadFromFile(const std::string& path);

            RigidTransformConfig left_camera;
            RigidTransformConfig lidar;
    };
}