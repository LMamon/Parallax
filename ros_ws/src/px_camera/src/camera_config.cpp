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
        fps = params["fps"].as<int>();

        exposure = params["exposure"].as<int>();
        gain = params["gain"].as<int>();

        return true;
    }

    catch (const YAML::Exception& e) {
        std::cerr << "Failed to load camera config: " << e.what() << '\n';
        return false;
    }
}