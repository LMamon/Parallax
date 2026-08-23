#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/visualization/video_encoder.hpp>
#include <parallax/stereo/calibration.hpp>
#include <parallax/pose/charuco_pose.hpp>

#include <foxglove/websocket.hpp>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::visualization {
    class Publisher{ 
        public:
            Publisher() = default;
            ~Publisher();

            Publisher(const Publisher&) = delete;
            Publisher& operator=(const Publisher&) = delete;

            bool initialize(std::uint32_t width, std::uint32_t height, std::uint32_t fps);

            bool publishLeftImage(const parallax::isp::RectifiedStereoFrame& frame, 
                                  const parallax::pose::CharucoPoseResult& pose,
                                  std::chrono::steady_clock::time_point timestamp);
            
            bool publishDepth(const parallax::isp::DepthFrame& frame);
                                
            bool publishLeftCalibration(const parallax::stereo::StereoCalibration& calibration);
            bool publishDisparity(const parallax::isp::StereoMatchFrame& frame);
            bool publishConfidence(const parallax::isp::StereoMatchFrame& frame);

            void shutdown();

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

        private:
            VideoEncoder video_encoder_;
            
            std::optional<foxglove::messages::CompressedVideoChannel> left_image_channel_;
            std::optional<foxglove::messages::CameraCalibrationChannel> left_calibration_channel_;
            std::optional<foxglove::messages::RawImageChannel> disparity_channel_;
            std::optional<foxglove::messages::RawImageChannel> confidence_channel_;
            std::optional<foxglove::messages::RawImageChannel> depth_channel_;
            
            cudaStream_t stream_ = nullptr;
            
            // Reusable pinned host staging.
            std::uint8_t* host_rgb_ = nullptr;
            std::int16_t* host_disparity_ = nullptr;
            std::uint16_t* host_confidence_ = nullptr;
            float* host_depth_ = nullptr;

            // Converted disparity for Foxglove 32FC1.
            std::vector<float> disparity_float_;

            // Reused H.264 output.
            std::vector<std::byte> encoded_video_;

            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;
            std::uint32_t fps_ = 0;


            bool initialized_ = false;
    };
}