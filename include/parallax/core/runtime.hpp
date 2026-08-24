#pragma once

#include <parallax/core/dependency_resolver.hpp>
#include <parallax/core/graph.hpp>

#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/stereo_camera.hpp>

#include <parallax/core/pipeline.hpp>
#include <parallax/core/sensor_frame.hpp>
#include <parallax/visualization/foxglove_server.hpp>
#include <parallax/visualization/publisher.hpp>

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
            
            [[nodiscard]] Graph& graph() noexcept { return graph_; }
            [[nodiscard]] const Graph& graph() const noexcept { return graph_; }
            [[nodiscard]] DependencyResolver& resolver() noexcept { return resolver_; }
            [[nodiscard]] const DependencyResolver& resolver() const noexcept { return resolver_; }

        private:
            parallax::camera::CameraConfig config_{};
            
            std::unique_ptr<parallax::camera::StereoCamera> camera_;
            std::atomic_bool running_{false};
            
            // Phase 4 graph infrastructure is owned by Runtime but remains
            // descriptive only. Legacy Pipeline continues to execute frames
            // until producers are migrated in later phases.
            Graph graph_;
            DependencyResolver resolver_{graph_};

            Pipeline pipeline_;
            bool initialized_ = false;
            
            parallax::visualization::FoxgloveServer foxglove_;
            parallax::visualization::Publisher publisher_;
        };
}