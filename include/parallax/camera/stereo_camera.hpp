#pragma once

#include <parallax/camera/arducam_device.hpp>
#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/frame_types.hpp>

#include <cstdint>
#include <memory>

namespace parallax::camera {

    struct ControlSetting {
            std::uint32_t id;
            std::int32_t value;
            const char* name;
        };

    class StereoCamera {
        public:
            explicit StereoCamera(CameraConfig config);
            ~StereoCamera();

            StereoCamera(const StereoCamera&) = delete;
            StereoCamera& operator=(const StereoCamera&) = delete;

            StereoCamera(StereoCamera&&) = delete;
            StereoCamera& operator=(StereoCamera&&) = delete;

            bool initialize();
            void shutdown();

            [[nodiscard]] bool isInitialized() const noexcept;
            [[nodiscard]] bool isStreaming() const noexcept;

            bool capture(RawFrame& frame);
            bool release(const RawFrame& frame);
            // bool warmup(std::uint32_t frame_count = 2);
            bool warmup();
            bool setControl(std::uint32_t control_id, std::int32_t value);

            [[nodiscard]] const CameraConfig& config() const noexcept;

        private:
            bool configureFormat();
            bool configureControls();

            CameraConfig config_;
            std::unique_ptr<ArducamDevice> device_;

            bool initialized_ = false;
    };

} // namespace parallax::camera