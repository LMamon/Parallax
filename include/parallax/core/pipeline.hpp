#pragma once

#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>

#include <parallax/core/sensor_frame.hpp>
#include <parallax/isp/isp.hpp>

#include <parallax/stereo/calibration.hpp>
#include <parallax/stereo/rectification.hpp>
#include <parallax/stereo/matcher.hpp>

#include <parallax/vpi/stream.hpp>

#include <filesystem>
#include <cstdint>

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

            void shutdown();
            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

        private: 
            parallax::isp::ISP isp_;

            parallax::stereo::StereoCalibration calibration_;
            parallax::vpi::Stream vpi_stream_;
            parallax::stereo::StereoRectifier rectifier_;
            parallax::stereo::StereoMatcher matcher_;

            std::uint64_t sequence_ = 0;
            bool initialized_ = false;
    };
}