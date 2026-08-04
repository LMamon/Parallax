#include <parallax/camera/camera_config.hpp>

#include <yaml-cpp/yaml.h>

#include <iostream>

namespace parallax::camera {

    bool CameraConfig::loadFromFile(const std::string& path) {
        try {
            YAML::Node config = YAML::LoadFile(path);

            device = config["device"].as<std::string>();
            frame_id = config["frame_id"].as<std::string>();

            width = config["width"].as<int>();
            height = config["height"].as<int>();
            frame_rate = config["frame_rate"].as<int>();

            exposure = config["exposure"].as<int>();
            analogue_gain = config["analogue_gain"].as<int>();

            trigger_mode = config["trigger_mode"].as<int>();
            disable_frame_timeout = config["disable_frame_timeout"].as<int>();
            frame_timeout = config["frame_timeout"].as<int>();

            horizontal_flip = config["horizontal_flip"].as<int>();
            vertical_flip = config["vertical_flip"].as<int>();

            return true;
        }
        catch (const YAML::Exception& e) {
            std::cerr
                << "Failed to load camera configuration: "
                << e.what()
                << '\n';

            return false;
        }
    }

} // namespace parallax::camera