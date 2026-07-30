#pragma once

#include "px_camera/arducam_device.hpp"

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

namespace px_camera {
    class PxCameraNode : public rclcpp::Node {
        public:
            explicit PxCameraNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

        private: 
            void loadParameters();

            std::string device_;
            std::string frame_id_;

            int width_;
            int height_;
            int fps_;
            int exposure_;
            int gain_;

            std::unique_ptr<ArducamDevice> camera_;
    };
}