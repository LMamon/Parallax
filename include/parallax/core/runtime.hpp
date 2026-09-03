#pragma once


#include <parallax/camera/camera_producer.hpp>
#include <parallax/camera/camera_config.hpp>
#include <parallax/camera/stereo_camera.hpp>

#include <parallax/core/dependency_resolver.hpp>
#include <parallax/application/request_controller.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/core/execution_gate.hpp>
#include <parallax/core/pipeline.hpp>
#include <parallax/core/sensor_frame.hpp>
#include <parallax/core/sensor_extrinsics.hpp>
#include <parallax/core/graph.hpp>

#include <parallax/visualization/foxglove_server.hpp>
#include <parallax/visualization/publisher.hpp>

#include <parallax/isp/isp_producer.hpp>
#include <parallax/core/execution_stats.hpp>

#include <parallax/stereo/rectification_producer.hpp>
#include <parallax/stereo/stereo_producer.hpp>
#include <parallax/stereo/depth_producer.hpp>

#include <parallax/perception/nanoowl_bridge.hpp>
#include <parallax/perception/detection_producer.hpp>
#include <parallax/perception/efficientvit_sam.hpp>
#include <parallax/perception/segmentation_producer.hpp>

#include <parallax/pose/charuco_pose_producer.hpp>
#include <parallax/pose/marker_depth_producer.hpp>

#include <parallax/lidar/rplidar.hpp>
#include <parallax/lidar/rplidar_producer.hpp>

#include <csignal>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <memory>
#include <thread>
#include <unordered_map>

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
                            const std::filesystem::path& sensor_extrinsics_path,
                            const std::filesystem::path& calibration_directory,
                            const std::filesystem::path& nanoowl_engine_path);
            
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
            std::unordered_map<const Producer*, ProducerExecutionState> producer_execution_state_;
            std::unordered_map<const Producer*, ProducerExecutionStats> producer_execution_stats_;
            std::chrono::steady_clock::time_point last_telemetry_publish_{};
            
            void runLidarSource();
            parallax::camera::CameraConfig config_{};
            SensorExtrinsics sensor_extrinsics_{};

            std::unique_ptr<parallax::camera::StereoCamera> camera_;
            std::atomic_bool running_{false};
            std::thread lidar_thread_;
            
            std::unique_ptr<parallax::perception::EfficientVitSam> efficientvit_sam_;
            std::unique_ptr<parallax::perception::SegmentationProducer> segmentation_producer_;

            /**
             * Runtime-scoped shared execution infrastructure.
             *
             * ExecutionContext owns completed graph products and the accelerator/CPU
             * execution lanes introduced in Phase 6. Runtime owns the context so its
             * lifetime encloses all producer submission.
             */
            ExecutionContext context_;
            
            // Producers are owned by Runtime because Graph stores non-owning Producer
            // pointers. Pipeline continues to own the underlying processing resources.
            //
            // This separates orchestration ownership from algorithm/resource ownership:
            // Runtime decides what runs; the existing implementation classes still own
            // their proven CUDA/VPI/OpenCV resources.
            std::unique_ptr<parallax::camera::CameraProducer> camera_producer_;
            std::unique_ptr<parallax::isp::IspProducer> isp_producer_;

            std::unique_ptr<parallax::stereo::RectificationProducer>rectification_producer_;

            std::unique_ptr<parallax::stereo::StereoProducer> stereo_producer_;
            std::unique_ptr<parallax::stereo::DepthProducer> depth_producer_;

            std::unique_ptr<parallax::pose::CharucoPoseProducer>charuco_pose_producer_;
            std::unique_ptr<parallax::pose::MarkerDepthPoducer>marker_depth_producer_;
            std::unique_ptr<parallax::perception::NanoOwlBridge> nanoowl_;
            std::unique_ptr<parallax::perception::DetectionProducer> detection_producer_;
            /**
            * LiDAR is Runtime-owned because its connection and scan lifecycle span the
            * full application runtime. Its producer is graph-visible but does not belong
            * to the camera-rate execution path.
            */
            std::unique_ptr<parallax::lidar::Rplidar> lidar_;
            std::unique_ptr<parallax::lidar::RplidarSourceProducer> lidar_producer_;

            Graph graph_;
            DependencyResolver resolver_{graph_};
            
            parallax::application::RequestController request_controller_{resolver_};
            Pipeline pipeline_;
            
            bool initialized_ = false;
            
            parallax::visualization::FoxgloveServer foxglove_;
            parallax::visualization::Publisher publisher_;
        };
}