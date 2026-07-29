#pragma once

#include <cstdint>
#include <string>

class CameraConfig {
    public:
        bool load(const std::string& path);

        std::string device;
        std::string frame_id;

        int width;
        int height;
        int fps;

        int exposure;
        int gain;
};