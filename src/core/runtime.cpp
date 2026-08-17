#include <parallax/core/runtime.hpp>

#include <iostream>

namespace parallax::core {
    Runtime::~Runtime() { shutdown(); }


    bool Runtime::initialize(const std::filesystem::path& camera_config_path,
                            const std::filesystem::path& calibration_directory) {

        if (initialized_) return true;
        if (!config_.loadFromFile(camera_config_path)) {
            std::cerr << "Runtime: Failed to load camera config\n";
            return false;
        }

        camera_ = std::make_unique<parallax::camera::StereoCamera>(config_);
        if (!camera_->initialize()) {
            std::cerr << "Runtime: failed to initialize camera\n";
            shutdown();
            return false;
        }

        if (!pipeline_.initialize(config_, calibration_directory)) {
            std::cerr << "Runtime: failed to initialize processing pipeline";
            shutdown();
            return true;
        }

        initialized_ = true;
        return true;
    }

    void Runtime::run(const volatile std::sig_atomic_t& stop_requested) {
        if (!initialized_) return;
        running_.store(true);

        parallax::camera::RawFrame raw_frame{};
        SensorFrame sensor_frame{};

        int failed_frames = 0;

        while (running_.load() && !stop_requested) {
            if (!camera_->capture(raw_frame)) {
                if (++failed_frames >= 10) {
                    std::cerr << "Runtime: camera failed to produce a valid frame\n";
                    break;
                }
                continue;
            }
            
            failed_frames = 0;
            if (!pipeline_.process(raw_frame, sensor_frame)) {
                std::cerr << "Runtime: pipeline processing failed\n";
                camera_->release(raw_frame);
                break;
            }

            if (!camera_->release(raw_frame)) {
                std::cerr << "Runtime: failed to release camera frame\n";
                break;
            }

            // processCommands();
            // dispatch(sensor_frame);
        }

        running_.store(false);
    }

    void Runtime::stop() noexcept {
        running_.store(false);
    }

    void Runtime::shutdown() {
        stop();
        pipeline_.shutdown();

        if (camera_) {
            camera_->shutdown();
            camera_.reset();
        }
        initialized_ = false;
    }
}