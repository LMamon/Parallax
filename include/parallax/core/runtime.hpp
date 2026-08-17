#pragma once

#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/stereo_camera.hpp>

#include <parallax/core/pipeline.hpp>
#include <parallax/core/sensor_frame.hpp>

#include <csignal>
#include <atomic>
#include <filesystem>
#include <memory>

namespace parallax::core {
    class Runtime {
        public:
            Runtime() = default;
            ~Runtime();

            Runtime(const Runtime&) = delete;
            Runtime& operator=(const Runtime&) = delete;

            Runtime(Runtime&&) = delete;
            Runtime& operator=(Runtime&&) = delete;

            bool initialize(const std::filesystem::path& camera_config_path,
                            const std::filesystem::path& calibration_directory);
            
            void run(const volatile std::sig_atomic_t& stop_requested);
            void stop() noexcept;
            void shutdown();

            [[nodiscard]] bool initialized() const noexcept { return initialized_; }

            [[nodiscard]] bool running() const noexcept { return running_.load(); }

        private:
            parallax::camera::CameraConfig config_{};
            std::unique_ptr<parallax::camera::StereoCamera> camera_;
            Pipeline pipeline_;
            std::atomic_bool running_{false};
            bool initialized_ = false;
        };
}