#pragma once

#include <parallax/camera/pixel_formats.hpp>
#include <string>

namespace parallax::camera {

    class CameraConfig {
        public:
            bool loadFromFile(const std::string& path);

            BayerPattern bayer_pattern = BayerPattern::GRBG;
            std::string device;
            std::string frame_id;

            int width = 0;
            int height = 0;
            int frame_rate = 0;

            int exposure = 0;
            int analogue_gain = 0;

            int trigger_mode = 0;
            int disable_frame_timeout = 0;
            int frame_timeout = 0;

            int horizontal_flip = 0;
            int vertical_flip = 0;
        };

} // namespace parallax::camera