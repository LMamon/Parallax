#include <px_camera/px_camera_node.hpp>

#include <memory>

namespace px_camera {
    PxCameraNode::PxCameraNode(const rclcpp::NodeOptions& options) : Node("px_camera", options) {
        //declaring parameters for ROS to populate them from YAML
        declare_parameter<std::string>("device", "/dev/video0");
        declare_parameter<std::string>("frame_id", "camera");

        declare_parameter<int>("width", 1920);
        declare_parameter<int>("height", 1200);
        declare_parameter<int>("fps", 30);

        declare_parameter<int>("exposure", 1000);
        declare_parameter<int>("gain", 4);

        loadParameters();

        //ROS node owns hardware interface
        camera_ = std::make_unique<ArducamDevice>(device_);

        RCLCPP_INFO(get_logger(),
                    "Configured camera %s: %dx%d @ %d FPS",
                    device_.c_str(),
                    width_,
                    height_,
                    fps_);
    }

    void PxCameraNode::loadParameters() {
        device_ = get_parameter("device").as_string();
        frame_id_ = get_parameter("frame_id").as_string();

        width_ = get_parameter("width").as_int();
        height_ = get_parameter("height").as_int();
        fps_ = get_parameter("fps").as_int();

        exposure_ = get_parameter("exposure").as_int();
        gain_ = get_parameter("gain").as_int();
    }

}