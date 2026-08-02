#pragma once

#include "px_camera/arducam_device.hpp"
#include "px_camera/camera_config.hpp"

#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <unordered_map>
#include <functional>

#include <rcl_interfaces/msg/set_parameters_result.hpp>

namespace px_camera {
    class PxCameraNode : public rclcpp::Node {
        ~PxCameraNode() override;

        public:
            explicit PxCameraNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

        private: 
            void loadParameters();

            bool initializeCamera();

            void streamingLoop();
            void publishRaw(const RawFrame& frame);

            CameraConfig config_;

            std::unique_ptr<ArducamDevice> camera_;
            rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr stereo_raw_pub_;
            rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;

            rcl_interfaces::msg::SetParametersResult
                parameterCallback(const std::vector<rclcpp::Parameter>& parameters);
            const std::unordered_map<std::string, uint32_t> control_ids_{{"exposure", arducam::EXPOSURE},
                                                                        {"analogue_gain", arducam::ANALOG_GAIN},
                                                                        {"frame_rate", arducam::FRAME_RATE},
                                                                        {"trigger_mode", arducam::TRIGGER_MODE},
                                                                        {"disable_frame_timeout", arducam::DISABLE_FRAME_TIMEOUT},
                                                                        {"frame_timeout", arducam::FRAME_TIMEOUT},
                                                                        {"horizontal_flip", arducam::HORIZONTAL_FLIP},
                                                                        {"vertical_flip", arducam::VERTICAL_FLIP},
            };

            const std::unordered_map<std::string, int CameraConfig::*> parameter_fields_{{"exposure", &CameraConfig::exposure},
                                                                    {"analogue_gain", &CameraConfig::analogue_gain},
                                                                    {"frame_rate", &CameraConfig::frame_rate},
                                                                    {"trigger_mode", &CameraConfig::trigger_mode},
                                                                    {"disable_frame_timeout", &CameraConfig::disable_frame_timeout},
                                                                    {"frame_timeout", &CameraConfig::frame_timeout},
                                                                    {"horizontal_flip", &CameraConfig::horizontal_flip},
                                                                    {"vertical_flip", &CameraConfig::vertical_flip},
                                                                };
 
            std::thread capture_thread_;
            std::atomic<bool> running_{false};
    };
}