#include <parallax/core/sensor_extrinsics.hpp>

#include <yaml-cpp/yaml.h>

#include <iostream>

namespace parallax::core {

    namespace {

        bool loadTransform(const YAML::Node& node,
                           const std::string& parent_frame,
                           const std::string& child_frame,
                           RigidTransformConfig& output) {

            if (!node) return false;

            output.parent_frame = parent_frame;
            output.child_frame = child_frame;

            const auto translation = node["translation_m"];
            output.translation_m = {translation["x"].as<double>(),
                                    translation["y"].as<double>(),
                                    translation["z"].as<double>()};

            const auto rotation = node["rotation_xyzw"];
            output.rotation_xyzw = {rotation["x"].as<double>(),
                                    rotation["y"].as<double>(),
                                    rotation["z"].as<double>(),
                                    rotation["w"].as<double>()};

            return true;
        }

    }

    bool SensorExtrinsics::loadFromFile(
        const std::string& path) {

            try {
                const YAML::Node config = YAML::LoadFile(path);
                const auto frames = config["frames"];

                const std::string body_frame = frames["body"].as<std::string>();
                if (!loadTransform(config["left_camera_from_body"],
                                   body_frame,
                                   frames["left_camera"].as<std::string>(),
                                   left_camera)) {

                    std::cerr << "Missing left-camera extrinsics\n";
                    return false;
                }

                if (!loadTransform(config["lidar_from_body"],
                                   body_frame,
                                   frames["lidar"].as<std::string>(),
                                   lidar)) {

                    std::cerr << "Missing LiDAR extrinsics\n";
                    return false;
                }

                return true;
            }
            catch (const YAML::Exception& e) {
                std::cerr << "Failed to load sensor extrinsics: " << e.what() << '\n';
                return false;
            }
    }

}