#include "px_isp/isp_node.hpp"

#include <functional>
#include <stdexcept>

namespace px_isp {
    
IspNode::IspNode(const rclcpp::NodeOptions& options) : rclcpp::Node("px_isp", options) {
    raw_sub_ = create_subscription<sensor_msgs::msg::Image>("stereo/raw",
                                                            rclcpp::SensorDataQoS(),
                                                            std::bind(&IspNode::imageCallback,
                                                                        this,
                                                                        std::placeholders::_1));

    rgb_pub_ = create_publisher<sensor_msgs::msg::Image>("stereo/rgb", rclcpp::SensorDataQoS());
    
    if (cudaStreamCreate(&cuda_stream_) != cudaSuccess) throw std::runtime_error("Failed to create CUDA stream.");
    RCLCPP_INFO(get_logger(), "px_isp started");
}
    
void IspNode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (msg->width == 0 || msg->height == 0 || msg->width % 2 != 0) {
        RCLCPP_ERROR(get_logger(),
                    "Invalid stereo dimensions: %ux%u",
                    msg->width,
                    msg->height);

        return;
    }

    const std::uint32_t eye_width = msg->width / 2;

    // Allocate persistent GPU buffers on the first frame.
    if (!stereo_gpu_.isAllocated()) {
        if (!stereo_gpu_.allocate(msg->width,
                                msg->height,
                                1,
                                sizeof(std::uint16_t))) {

            RCLCPP_ERROR(get_logger(), "Failed to allocate stereo GPU buffer.");
            return;
        }

        if (!left_raw_.allocate(eye_width,
                            msg->height,
                            1,
                            sizeof(std::uint16_t))) {

            RCLCPP_ERROR(get_logger(), "Failed to allocate left RAW GPU buffer.");
            return;
        }

        if (!right_raw_.allocate(eye_width,
                                msg->height,
                                1,
                                sizeof(std::uint16_t))) {

            RCLCPP_ERROR(get_logger(), "Failed to allocate right RAW GPU buffer.");
            return;
        }

        if (!left_rgb_.allocate(eye_width,
                            msg->height,
                            3,
                            sizeof(std::uint16_t))) {

            RCLCPP_ERROR(get_logger(), "Failed to allocate left RGB GPU buffer.");
            return;
        }

        if (!right_rgb_.allocate(eye_width,
                                msg->height,
                                3,
                                sizeof(std::uint16_t))) {

            RCLCPP_ERROR(get_logger(), "Failed to allocate right RGB GPU buffer.");
            return;
        }

        RCLCPP_INFO(get_logger(),
                    "Allocated ISP buffers for %ux%u stereo input.",
                    msg->width,
                    msg->height);
    }

    // CPU ROS image → GPU.
    if (!stereo_gpu_.upload(msg->data.data(), msg->step, cuda_stream_)) {
        RCLCPP_ERROR(get_logger(), "Failed to upload stereo image to GPU.");
        return;
    }

    // GPU stereo image → left and right RAW GPU buffers.
    if (!splitter_.process(stereo_gpu_,
                        left_raw_,
                        right_raw_,
                        cuda_stream_)) {
        RCLCPP_ERROR(get_logger(), "Failed to split stereo GPU image.");
        return;
    }

    // Remaining ISP stages:
    //
    // demosaic_.process(left_raw_, right_raw_,
    //                   left_rgb_, right_rgb_,
    //                   cuda_stream_);
    //
    // black_level_.process(left_rgb_, right_rgb_, cuda_stream_);
    // white_balance_.process(left_rgb_, right_rgb_, cuda_stream_);
    // color_correction_.process(left_rgb_, right_rgb_, cuda_stream_);
    // gamma_.process(left_rgb_, right_rgb_, cuda_stream_);

    // No RGB publish yet. The output remains GPU-resident until the
    // demosaic and NITROS publishing path are implemented.
}

IspNode::~IspNode() {
    if (cuda_stream_) {
        cudaStreamSynchronize(cuda_stream_);
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
}

} // namespace px_isp 