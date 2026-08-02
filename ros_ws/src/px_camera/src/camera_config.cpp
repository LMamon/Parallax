#include "px_camera/camera_config.hpp"

#include <yaml-cpp/yaml.h>
#include <iostream>

bool CameraConfig::load(const std::string& path) {
    try {
        YAML::Node r = YAML::LoadFile(path);
        
        const auto params = r["/**"]["ros__parameters"];

        if (!params) {
            std::cerr <<"Missing /**/ros__parameters in: " << path << "\n";
            return false;
        }

        device = params["device"].as<std::string>();
        frame_id = params["frame_id"].as<std::string>();

        width = params["width"].as<int>();
        height = params["height"].as<int>();
        frame_rate = params["frame_rate"].as<int>();

        exposure = params["exposure"].as<int>();

        analogue_gain = params["analogue_gain"].as<int>();
        trigger_mode = params["trigger_mode"].as<int>();
        disable_frame_timeout = params["disable_frame_timeout"].as<int>();

        frame_timeout = params["frame_timeout"].as<int>();

        horizontal_flip = params["horizontal_flip"].as<int>();
        vertical_flip = params["vertical_flip"].as<int>();

        return true;
    }

    catch (const YAML::Exception& e) {
        std::cerr << "Failed to load camera config: " << e.what() << '\n';
        return false;
    }
}