#include <parallax/camera/stereo_camera.hpp>
#include <parallax/camera/pixel_formats.hpp>
#include <parallax/camera/arducam_controls.hpp>
#include <parallax/camera/logger.hpp>
#include <parallax/camera/arducam_controls.hpp>

#include <linux/videodev2.h>
#include <iostream>
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
        
             // NOTE:
        // The Jetson/Arducam driver resets several sensor controls during
        // VIDIOC_STREAMON, so controls are applied after startStreaming().
        // Do not move configureControls() before STREAMON unless the driver
        // behavior changes.

        if (!device_->initializeStreaming()) {
            shutdown();
            return false;
        }

        if (!device_->startStreaming()) {
            shutdown();
            return false;
        }
        /*
        * The Arducam/Jetson driver resets exposure, gain, and frame rate
        * to defaults during STREAMON. Reapply the YAML configuration after
        * streaming has started.
        */
        if (!configureControls()) {
            shutdown();
            return false;
        }
        initialized_ = true;

        if (!warmup()) {
            shutdown();
            return false;
        }

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

    bool StereoCamera::capture(RawFrame& frame) {
        if (!initialized_ || !device_->isStreaming()) {
            logMessage("StereoCamera::capture: camera is not streaming");
            return false;
        }

        return device_->dequeue(frame);
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
        if (!device_->setFormat(static_cast<std::uint32_t>(config_.width),
                                static_cast<std::uint32_t>(config_.height),
                                V4L2_PIX_FMT_BA10)) {

            return false;
        }

        // const auto format = device_->getPixelFormat();
        const std::uint32_t actual_format = device_->getPixelFormat();
        if (actual_format == 0) return false;

        char fourcc[5] = {static_cast<char>(actual_format & 0xff),
                        static_cast<char>((actual_format >> 8) & 0xff),
                        static_cast<char>((actual_format >> 16) & 0xff),
                        static_cast<char>((actual_format >> 24) & 0xff),
                        '\0'};

        std::cout
            << "Camera: "
            << config_.width << "x" << config_.height
            << "  Pixel Format: "
            << fourcc
            << " (0x"
            << std::hex << actual_format << std::dec << ")\n";

        // return format != 0;
        return actual_format == V4L2_PIX_FMT_BA10;
    }

    bool StereoCamera::configureControls() {
        const ControlSetting settings[] = {{controls::AnalogGain,
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
                                            "vertical_flip"},
                                        {controls::Exposure,
                                            static_cast<std::int32_t>(config_.exposure),
                                            "exposure"}
        };

        for (const auto& setting : settings) {
            if (!device_->setControl(setting.id, setting.value)) {
                logError("Failed to configure camera control", setting.name);
                return false;
            }
            int32_t actual{};
            if (device_->getControl(setting.id, actual)) {

                std::cout
                    << setting.name
                    << ": requested " << setting.value
                    << " actual " << actual
                    << '\n';
            }
        }
        return true;
    }

    bool StereoCamera::warmup() {
        using Clock = std::chrono::steady_clock;
        using Milliseconds = std::chrono::milliseconds;

        RawFrame frame{};
        const std::uint32_t frame_rate = std::max<std::uint32_t>(config_.frame_rate, 1);
        const std::uint32_t frame_period_ms = (1000U + frame_rate - 1U) / frame_rate;
        const std::uint32_t exposure_ms = (config_.exposure + 999U) / 1000U;

        /*
        * Startup may include:
        * - driver frame timeout/reset behavior
        * - control settling after STREAMON
        * - several invalid startup frames
        *
        * Use two configured frame-timeout windows, while also ensuring
        * exposure and several frame periods are covered.
        */

        const std::uint32_t startup_timeout_ms = std::max(2U * config_.frame_timeout,
                                                        config_.frame_timeout + exposure_ms + 4U * frame_period_ms);

        const auto deadline = Clock::now() + Milliseconds(startup_timeout_ms);
        std::uint32_t attempts = 0;

        while (Clock::now() < deadline) {
            ++attempts;
            if (!capture(frame)) continue;

            if (!release(frame)) {
                logMessage("StereoCamera::warmup: failed to release startup frame");
                return false;
            }

            if (attempts > 1) {
                std::cout
                    << "Camera warmup completed after "
                    << attempts
                    << " capture attempt(s)\n";
            }
            return true;
        }

        std::cout
            << "Camera warmup timed out after "
            << startup_timeout_ms
            << " ms and "
            << attempts
            << " capture attempt(s)\n";

        return false;
    }
}