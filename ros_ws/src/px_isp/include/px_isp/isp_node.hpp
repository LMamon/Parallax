#pragma once

// #include "px_isp/black_level.cuh"
// #include "px_isp/demosaic.cuh"
// #include "px_isp/color_correction.cuh"
// #include "px_isp/gamma.cuh"
#include "px_isp/splitter.cuh"
// #include "px_isp/white_balance.cuh"

#include "px_isp/cuda_buffer.cuh"

#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace px_isp {
    class IspNode : public rclcpp::Node {
        public:
            explicit IspNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

        ~IspNode() override;

        private:
            void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
            rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_sub_;

            rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_;

            CudaBuffer stereo_gpu_;
            cudaStream_t cuda_stream_ = nullptr;

            CudaBuffer left_raw_;
            CudaBuffer right_raw_;
            CudaBuffer left_rgb_;
            CudaBuffer right_rgb_;

            Splitter splitter_;
            // Demosaic demosaic_;
            // BlackLevel black_level_;
            // WhiteBalance white_balance_;
            // ColorCorrection color_correction_;
            // Gamma gamma_;
    };
}