#include <px_camera/px_camera_node.hpp>
#include <px_camera/logger.hpp>

#include <memory>
#include <string>
#include <cstring>
#include <rclcpp/rclcpp.hpp>

namespace arducam {
    constexpr uint32_t EXPOSURE = 0x00980911;
    constexpr uint32_t HORIZONTAL_FLIP = 0x00980914;
    constexpr uint32_t VERTICAL_FLIP = 0x00980915;
    constexpr uint32_t TRIGGER_MODE = 0x00981901;
    constexpr uint32_t DISABLE_FRAME_TIMEOUT = 0x00981902;
    constexpr uint32_t FRAME_TIMEOUT = 0x00981903;
    constexpr uint32_t FRAME_RATE = 0x00981906;
    constexpr uint32_t ANALOG_GAIN = 0x009e0903;
}

namespace px_camera {
    PxCameraNode::PxCameraNode(const rclcpp::NodeOptions& options) : Node("px_camera", options) {
        //declaring parameters for ROS to populate them from YAML
        declare_parameter<std::string>("device", "/dev/video0");
        declare_parameter<std::string>("frame_id", "camera");

        declare_parameter<int>("width", 3840);
        declare_parameter<int>("height", 1200);

        declare_parameter<int>("frame_rate", 38);

        declare_parameter<int>("exposure", 1000);
        declare_parameter<int>("analogue_gain", 100);

        declare_parameter<int>("trigger_mode", 0);
        declare_parameter<int>("disable_frame_timeout", 0);
        declare_parameter<int>("frame_timeout", 2000);

        declare_parameter<int>("horizontal_flip", 0);
        declare_parameter<int>("vertical_flip", 0);

        loadParameters();

        //ROS node owns hardware interface
        camera_ = std::make_unique<ArducamDevice>(config_.device);
        if (!initializeCamera()) throw std::runtime_error("Failed to initialize camera.");

        stereo_raw_pub_ = create_publisher<sensor_msgs::msg::Image>("stereo/raw",
                                                                    rclcpp::SensorDataQoS());
        running_ = true;
        capture_thread_ = std::thread(&PxCameraNode::streamingLoop, this);

        RCLCPP_INFO(get_logger(),
                    "Configured camera %s: %dx%d @ %d FPS",
                    config_.device.c_str(),
                    config_.width,
                    config_.height,
                    config_.frame_rate);

        parameter_callback_ = add_on_set_parameters_callback(std::bind(&PxCameraNode::parameterCallback,
                                                                        this,
                                                                        std::placeholders::_1));
    }

    void PxCameraNode::loadParameters() {
        config_.device = get_parameter("device").as_string();
        config_.frame_id = get_parameter("frame_id").as_string();

        config_.width = get_parameter("width").as_int();
        config_.height = get_parameter("height").as_int();

        config_.frame_rate = get_parameter("frame_rate").as_int();

        config_.exposure = get_parameter("exposure").as_int();
        config_.analogue_gain = get_parameter("analogue_gain").as_int();

        config_.trigger_mode = get_parameter("trigger_mode").as_int();
        config_.disable_frame_timeout = get_parameter("disable_frame_timeout").as_int();

        config_.frame_timeout = get_parameter("frame_timeout").as_int();

        config_.horizontal_flip = get_parameter("horizontal_flip").as_int();
        config_.vertical_flip = get_parameter("vertical_flip").as_int();
    }
    

    bool PxCameraNode::initializeCamera() {
        if (!camera_->open()) return false;
        if (!camera_->setFormat(config_.width, config_.height, V4L2_PIX_FMT_SRGGB10)) return false;

        if (!camera_->setControl(arducam::EXPOSURE, config_.exposure)) return false;
        if (!camera_->setControl(arducam::ANALOG_GAIN, config_.analogue_gain)) return false;
        if (!camera_->setControl(arducam::FRAME_RATE, config_.frame_rate)) return false;
        if (!camera_->setControl(arducam::TRIGGER_MODE, config_.trigger_mode)) return false;
        if (!camera_->setControl(arducam::DISABLE_FRAME_TIMEOUT, config_.disable_frame_timeout)) return false;
        if (!camera_->setControl(arducam::FRAME_TIMEOUT, config_.frame_timeout)) return false;
        if (!camera_->setControl(arducam::HORIZONTAL_FLIP, config_.horizontal_flip)) return false;
        if (!camera_->setControl(arducam::VERTICAL_FLIP, config_.vertical_flip)) return false;

        if (!camera_->initializeStreaming()) return false;
        if (!camera_->startStreaming()) return false;

        return true;
    }

    void PxCameraNode::streamingLoop() {
        while (running_) {
            RawFrame frame;

            if (!camera_->dequeue(frame)) continue;
            publishRaw(frame);

            if (!camera_->queue(frame)) {
                RCLCPP_ERROR(get_logger(), "Failed to requeue V4L2 buffer.");
                running_ = false;
                break;
            }
        }
    }

    void PxCameraNode::publishRaw(const RawFrame& frame) {
        sensor_msgs::msg::Image msg;

        msg.header.stamp = rclcpp::Time(frame.timestamp.count());
        msg.header.frame_id = config_.frame_id;

        msg.width = frame.width;
        msg.height = frame.height;

        msg.encoding = "mono16";

        msg.is_bigendian = false;
        msg.step = frame.width * sizeof(uint16_t);

        msg.data.resize(frame.bytes);

        std::memcpy(msg.data.data(),
                    frame.data,
                    frame.bytes);
                    
        stereo_raw_pub_->publish(msg);
    }

    PxCameraNode::~PxCameraNode() {
        running_ = false;
        if (capture_thread_.joinable()) capture_thread_.join();

        camera_->stopStreaming();
        camera_->shutdownStreaming();
        
        camera_.reset();
    }

    rcl_interfaces::msg::SetParametersResult PxCameraNode::parameterCallback(const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        for (const auto& parameter : parameters) {
            auto control = control_ids_.find(parameter.get_name());
            auto field = parameter_fields_.find(parameter.get_name());

            if (control == control_ids_.end() || field == parameter_fields_.end()) continue;

            const int value = parameter.as_int();

            if (!camera_->setControl(control->second, value)) {
                result.successful = false;
                result.reason = "Failed to update " + parameter.get_name();
                return result;
            }

            config_.*(field->second) = value;
        }
        return result;
    }
}