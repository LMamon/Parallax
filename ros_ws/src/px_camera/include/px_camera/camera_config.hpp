#pragma once

#include <string>

class CameraConfig {
    public:
        bool load(const std::string& path);

        std::string device;
        std::string frame_id;

        int width;
        int height;
        int frame_rate;
        int exposure;
        int analogue_gain;
        int trigger_mode;
        int disable_frame_timeout;
        int frame_timeout;
        int horizontal_flip;
        int vertical_flip;
};