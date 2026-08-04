#include <parallax/camera/stereo_camera.hpp>

#include <parallax/camera/arducam_controls.hpp>
#include <parallax/camera/logger.hpp>

#include <linux/videodev2.h>

#include <utility>

namespace parallax::camera {

    StereoCamera::StereoCamera(CameraConfig config) : config_(std::move(config)), 
                                                    device_(std::make_unique<ArducamDevice>(config_.device)) {}

    StereoCamera::~StereoCamera() { 
        shutdown();
    }

    bool StereoCamera::initialize() {
        if (initialized_) return true;

        if (!device_) {
            logMessage("StereoCamera::initialize: camera device is unavailable");
            return false;
        }

        if (!device_->open()) {
            return false;
        }

        if (!configureFormat()) {
            shutdown();
            return false;
        }

        if (!configureControls()) {
            shutdown();
            return false;
        }

        if (!device_->initializeStreaming()) {
            shutdown();
            return false;
        }

        if (!device_->startStreaming()) {
            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    void StereoCamera::shutdown() {
        if (!device_) {
            initialized_ = false;
            return;
        }

        device_->stopStreaming();
        device_->shutdownStreaming();
        device_->close();

        initialized_ = false;
    }

    bool StereoCamera::isInitialized() const noexcept {
        return initialized_;
    }

    bool StereoCamera::isStreaming() const noexcept {
        return device_ != nullptr && device_->isStreaming();
    }

    bool StereoCamera::capture(RawFrame& frame, int timeout_ms) {
        if (!initialized_ || !device_->isStreaming()) {
            logMessage("StereoCamera::capture: camera is not streaming");
            return false;
        }

        return device_->dequeue(frame, timeout_ms);
    }

    bool StereoCamera::release(const RawFrame& frame) {
        if (!initialized_ || !device_->isStreaming()) {
            logMessage("StereoCamera::release: camera is not streaming");
            return false;
        }

        return device_->queue(frame);
    }

    bool StereoCamera::setControl(std::uint32_t control_id, std::int32_t value) {
        if (!device_ || !device_->isOpen()) {
            logMessage("StereoCamera::setControl: camera device is not open");
            return false;
        }

        return device_->setControl(control_id, value);
    }

    const CameraConfig& StereoCamera::config() const noexcept {
        return config_;
    }

    bool StereoCamera::configureFormat() {
        /*
        * The Arducam stereo device currently exposes the packed stereo Bayer
        * stream as 10-bit RGGB data.
        *
        * Keep this hardware format decision in the camera layer for now. If
        * additional sensors or pixel formats are introduced, add pixel_format
        * to CameraConfig instead of scattering format checks downstream.
        */
        return device_->setFormat(static_cast<std::uint32_t>(config_.width),
                                    static_cast<std::uint32_t>(config_.height),
                                    V4L2_PIX_FMT_SRGGB10);
    }

    bool StereoCamera::configureControls() {
        struct ControlSetting {
            std::uint32_t id;
            std::int32_t value;
            const char* name;
        };

        const ControlSetting settings[] = {{controls::Exposure,
                                            static_cast<std::int32_t>(config_.exposure),
                                            "exposure"},
                                        {controls::AnalogGain,
                                            static_cast<std::int32_t>(config_.analogue_gain),
                                            "analogue_gain"},
                                        {controls::FrameRate,
                                            static_cast<std::int32_t>(config_.frame_rate),
                                            "frame_rate"},
                                        {controls::TriggerMode,
                                            static_cast<std::int32_t>(config_.trigger_mode),
                                            "trigger_mode"},
                                        {controls::DisableFrameTimeout,
                                            static_cast<std::int32_t>(config_.disable_frame_timeout),
                                            "disable_frame_timeout"},
                                        {controls::FrameTimeout,
                                            static_cast<std::int32_t>(config_.frame_timeout),
                                            "frame_timeout"},
                                        {controls::HorizontalFlip,
                                            static_cast<std::int32_t>(config_.horizontal_flip),
                                            "horizontal_flip"},
                                        {controls::VerticalFlip,
                                            static_cast<std::int32_t>(config_.vertical_flip),
                                            "vertical_flip"}
        };

        for (const ControlSetting& setting : settings) {
            if (!device_->setControl(setting.id, setting.value)) {
                logError("Failed to configure camera control", setting.name);
                return false;
            }
        }
        return true;
    }
}