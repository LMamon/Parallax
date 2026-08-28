#pragma once

#include <parallax/isp/frame_types.hpp>
#include <parallax/visualization/video_encoder.hpp>
#include <parallax/stereo/calibration.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <parallax/visualization/foxglove_server.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/core/sensor_extrinsics.hpp>

#include <parallax/core/completion.hpp>
#include <parallax/lidar/frame_types.hpp>

#include <functional>
#include <foxglove/websocket.hpp>
#include <cuda_runtime.h>

#include <string>
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
            using HostWait = std::function<bool(const parallax::core::CompletionHandle&)>;
            bool initialize(FoxgloveServer& foxglove, std::uint32_t width, std::uint32_t height, std::uint32_t fps);

            /**
            * Observe the newest graph products currently available in ProductStore.
            *
            * Publication is downstream-only:
            * - no producer execution;
            * - no graph resolution;
            * - no demand acquisition;
            * - no visualization-only accelerator wait when a channel has no sinks.
            */

            bool publishLeftCalibration(const parallax::stereo::StereoCalibration& calibration);
            bool publishStaticTransforms(const parallax::core::SensorExtrinsics& extrinsics);
            bool publishAvailable(const parallax::core::ProductStore& store, const HostWait& wait_for_host);
            bool publishRuntimeTelemetry(const std::string& json);
            void shutdown();

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

        private:
            bool publishLeftImage(const parallax::isp::RectifiedStereoFrame& frame, 
                                  const parallax::pose::CharucoPoseResult* pose);
            
            bool publishDepth(const parallax::isp::DepthFrame& frame);
            bool publishDisparity(const parallax::isp::StereoMatchFrame& frame);
            bool publishLidarScan(const parallax::lidar::LidarScan& scan);

            VideoEncoder video_encoder_;
            cudaStream_t stream_ = nullptr;
            
            // Reusable pinned host staging.
            std::uint8_t* host_rgb_ = nullptr;
            std::int16_t* host_disparity_ = nullptr;
            float* host_depth_ = nullptr;

            // Converted disparity for Foxglove 32FC1.
            std::vector<float> disparity_float_;

            // Reused H.264 output.
            std::vector<std::byte> encoded_video_;
            std::uint32_t width_ = 0;
            std::uint32_t height_ = 0;
            std::uint32_t fps_ = 0;

            /**
             * Non-owning access to the native Foxglove capability surface.
             *
             * Runtime owns FoxgloveServer and shuts Publisher down before the server.
             * Publisher performs serialization/logging only; it does not define channel
             * lifetime or subscription semantics.
             */
            FoxgloveServer* foxglove_ = nullptr;
            bool initialized_ = false;
    };
}