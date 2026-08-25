#pragma once

#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>

#include <parallax/core/sensor_frame.hpp>
#include <parallax/isp/isp.hpp>
#include <parallax/pose/charuco_pose.hpp>

#include <parallax/stereo/calibration.hpp>
#include <parallax/stereo/rectification.hpp>
#include <parallax/stereo/matcher.hpp>

#include <parallax/vpi/stream.hpp>

#include <filesystem>
#include <cstdint>
#include <chrono>

namespace parallax::core {
    class Pipeline {
        public:
            Pipeline() = default;
            ~Pipeline();

            Pipeline(const Pipeline&) = delete;
            Pipeline& operator=(const Pipeline&) = delete;

            Pipeline(Pipeline&&) = delete;
            Pipeline& operator=(Pipeline&&) = delete;

            bool initialize(const parallax::camera::CameraConfig& config, const std::filesystem::path& calibration);
            bool process(const parallax::camera::RawFrame& input, SensorFrame& output);
            
            bool synchronize();
            const parallax::isp::RectifiedStereoFrame& rgb() const noexcept {
                return rectifier_.rgb();
            }
            
            const parallax::stereo::StereoCalibration& calibration() const noexcept {
                return calibration_;
            }

            /**
             * graph producers reuse the already-initialized processing resources
             * owned by Pipeline. Pipeline remains the lifetime/compatibility container while
             * Runtime takes over orchestration through the dependency graph.
             *
             * These accessors do not transfer ownership.
             */
            parallax::isp::ISP& isp() noexcept { return isp_; }
            parallax::isp::DepthFrame& depth() noexcept { return depth_; }
            parallax::vpi::Stream& vpiStream() noexcept { return vpi_stream_; }
            parallax::stereo::StereoRectifier& rectifier() noexcept { return rectifier_; }
            parallax::stereo::StereoMatcher& matcher() noexcept { return matcher_; }
            parallax::pose::CharucoPose& charucoPose() noexcept { return charuco_pose_; }

            void shutdown();
            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

        private: 
            parallax::isp::ISP isp_;
            parallax::isp::DepthFrame depth_;

            parallax::stereo::StereoCalibration calibration_;
            parallax::vpi::Stream vpi_stream_;
            parallax::stereo::StereoRectifier rectifier_;
            parallax::stereo::StereoMatcher matcher_;
            parallax::pose::CharucoPose charuco_pose_;

            std::chrono::steady_clock::time_point fps_window_start_{};
            std::uint32_t fps_frame_count_ = 0;
            double pipeline_fps_ = 0.0;


            std::uint32_t timing_frames = 0;
            std::uint64_t sequence_ = 0;
            bool initialized_ = false;
    };
}