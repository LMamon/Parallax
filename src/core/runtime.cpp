#include <parallax/core/runtime.hpp>
#include <parallax/core/pipeline.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <parallax/core/history_configuration.hpp>
#include <parallax/core/runtime_metrics.hpp>

#include <nlohmann/json.hpp>
#include <chrono>
#include <vector>
#include <iostream>

namespace parallax::core {
    Runtime::~Runtime() { shutdown(); }

    bool Runtime::initialize(const std::filesystem::path& camera_config_path,
                            const std::filesystem::path& sensor_extrinsics_path,
                             const std::filesystem::path& calibration_directory) {

        if (initialized_) return true;

        if (!context_.initialize()) {
            std::cerr << "Runtime: failed to initialize execution context\n";
            shutdown();
            return false;
        }

        if (!config_.loadFromFile(camera_config_path)) {
            std::cerr << "Runtime: failed to load camera config\n";
            return false;
        }

        if (!sensor_extrinsics_.loadFromFile(sensor_extrinsics_path)) {
            std::cerr << "Runtime: failed to load extrinsics config\n";
            return false;   
        }

        camera_ = std::make_unique<parallax::camera::StereoCamera>(config_);
        if (!camera_->initialize()) {
            std::cerr << "Runtime: failed to initialize camera\n";
            shutdown();
            return false;
        }

        lidar_ = std::make_unique<parallax::lidar::Rplidar>();
        if (!lidar_->initialize()) {
            std::cerr << "Runtime: failed to initialize RPLIDAR\n";
            shutdown();
            return false;
        }
        /**
         * Pipeline still owns the proven processing resources during Phase 5:
         * ISP allocations, VPI stream, rectifier, matcher, depth storage, and pose
         * estimator. Runtime now takes over orchestration through graph producers.
         */
        if (!pipeline_.initialize(config_, calibration_directory)) {
            std::cerr << "Runtime: failed to initialize processing pipeline\n";
            shutdown();
            return false;
        }

        /**
         * Construct the complete producer set before registering or finalizing the
         * graph. Graph stores non-owning Producer pointers, so Runtime owns every
         * producer for the lifetime of the graph.
         */
        camera_producer_ = std::make_unique<parallax::camera::CameraProducer>(*camera_, context_.products());
        lidar_producer_ = std::make_unique<parallax::lidar::RplidarSourceProducer>(*lidar_, context_.products());
        isp_producer_ = std::make_unique<parallax::isp::IspProducer>(pipeline_.isp(), context_.products());

        rectification_producer_ = std::make_unique<parallax::stereo::RectificationProducer>(
                                                   pipeline_.rectifier(),
                                                   pipeline_.calibration(),
                                                   context_.products());

        stereo_producer_ = std::make_unique<parallax::stereo::StereoProducer>(pipeline_.matcher(), context_.products());

        depth_producer_ = std::make_unique<parallax::stereo::DepthProducer>(pipeline_.calibration(),
                                                                            context_.products());

        charuco_pose_producer_ = std::make_unique<parallax::pose::CharucoPoseProducer>(
                                                  pipeline_.charucoPose(),
                                                  pipeline_.calibration(),
                                                  context_.products());

        marker_depth_producer_ = std::make_unique<parallax::pose::MarkerDepthPoducer>(context_.products());

        /**
         * Registration describes the complete concrete dependency graph.
         * Finalization happens exactly once, after every producer is present.
         */
        graph_.register_producer(*camera_producer_);
        graph_.register_producer(*isp_producer_);
        graph_.register_producer(*rectification_producer_);
        graph_.register_producer(*stereo_producer_);
        graph_.register_producer(*depth_producer_);
        graph_.register_producer(*charuco_pose_producer_);
        graph_.register_producer(*marker_depth_producer_);
        graph_.register_producer(*lidar_producer_);

        graph_.finalize();

        configure_ordered_history(graph_, context_.products());

        /**
         * Persistent perception-service demand.
         *
         * These are deliberately root products rather than an exhaustive list of
         * intermediates. DependencyResolver derives the required producer subgraph.
         *
         * RectifiedRgb keeps camera -> ISP -> rectification active.
         * Confidence extends that path through stereo matching.
         * MarkerDepth extends it through metric depth and marker pose/depth.
         * LidarScan keeps the independent RPLIDAR source active.
         *
         * Foxglove observation is not required for any of these products to exist.
         */
        resolver_.acquire(ProductId::RectifiedRgb, DemandSource::RuntimeBaseline);
        resolver_.acquire(ProductId::Disparity, DemandSource::RuntimeBaseline);
        resolver_.acquire(ProductId::MarkerDepth, DemandSource::RuntimeBaseline);
        resolver_.acquire(ProductId::LidarScan, DemandSource::RuntimeBaseline);


        parallax::visualization::FoxgloveServer::DemandCallbacks foxglove_demand;
        foxglove_demand.acquire = [this](ProductId product) {
                /**
                 * Subscription callbacks record external demand only.
                 * They deliberately do not resolve or submit the graph here.
                 */
                resolver_.acquire(product, DemandSource::FoxgloveSubscriber);
            };

        foxglove_demand.release = [this](ProductId product) { 
                                    resolver_.release(
                                    product,
                                    DemandSource::FoxgloveSubscriber);
            };

        if (!foxglove_.initialize(std::move(foxglove_demand))) {
            std::cerr << "Runtime: failed to initialize Foxglove server\n";
            shutdown();
            return false;
        }

        const auto& rgb = pipeline_.rgb();

        if (!publisher_.initialize(foxglove_, rgb.width, rgb.height, config_.frame_rate)) {
            std::cerr << "Runtime: failed to initialize visualization publisher\n";

            shutdown();
            return false;
        }
        
        initialized_ = true;
        return true;
    }

    void Runtime::run(const volatile std::sig_atomic_t& stop_requested) {
        if (!initialized_) return;
        running_.store(true);
        int failed_frames = 0;
        last_telemetry_publish_ = std::chrono::steady_clock::now();

        /**
        * Camera-domain baseline execution.
        *
        * RuntimeBaseline defines what the persistent perception service maintains;
        * Foxglove merely decides which of those products incur observation cost.
        *
        * LidarScan is intentionally excluded from this plan because its source has
        * an independent worker/cadence below.
        */

        const std::vector<ProductId> baseline_camera_products{
            ProductId::RectifiedRgb, ProductId::Disparity, ProductId::MarkerDepth
        };
        const auto execution_plan = resolver_.resolve(baseline_camera_products);

        /**
         * Populate the stats map before the LiDAR worker starts.
         *
         * The camera thread and LiDAR thread then mutate only their own already-created
         * ProducerExecutionStats entries. Neither thread inserts into the unordered_map
         * while the other is running.
         */
        for (Producer* producer : execution_plan) {
            if (producer != nullptr) {
                producer_execution_stats_.try_emplace(producer);
            }
        }

        if (lidar_producer_) {
            producer_execution_stats_.try_emplace(lidar_producer_.get());
        }

        lidar_thread_ = std::thread(&Runtime::runLidarSource, this);

        while (running_.load() && !stop_requested) {
            bool frame_failed = false;

            for (Producer* producer : execution_plan) {
                if (producer == nullptr) {
                    frame_failed = true;
                    break;
                }

                auto& stats = producer_execution_stats_.at(producer);
                ++stats.considered;

                const auto policy = producer->execution_policy();
                const auto input = input_observation_with_timestamp(*producer, context_.products());

                if (!producer->inputs().empty()) {
                    if (!input) {
                        ++stats.missing_or_incompatible_input;
                        frame_failed = true;
                        break;
                    }

                    auto& state = producer_execution_state_[producer];
                    const auto now = ExecutionContext::now();

                    switch (submission_decision(policy, state, *input, now)) {
                        case SubmissionDecision::Submit:
                            break;

                        case SubmissionDecision::StaleInput:
                            ++stats.stale_input;
                            continue;

                        case SubmissionDecision::RateLimited:
                            ++stats.rate_limited;
                            continue;

                        case SubmissionDecision::Superseded:
                            ++stats.superseded;
                            continue;
                    }
                }

                const SubmitResult result = producer->submit(context_);
                if (result == SubmitResult::Failed) {
                    ++stats.failed;
                    std::cerr << "Runtime: producer failed: " << producer->name() << '\n';
                    frame_failed = true;
                    break;
                }

                if (result == SubmitResult::NoWork) {
                    ++stats.no_work;
                    frame_failed = true;
                    break;
                }

                ++stats.submitted;

                if (input) {
                    record_submission(producer_execution_state_[producer], policy, *input, ExecutionContext::now());
                }
            }


            if (frame_failed) {
                if (++failed_frames >= 10) {
                    std::cerr << "Runtime: graph failed to produce a valid frame\n";
                    break;
                }
                continue;
            }

            failed_frames = 0;

            if (foxglove_.takeCalibrationRequest()) {
                if (!publisher_.publishLeftCalibration(pipeline_.calibration())) {
                    std::cerr << "Runtime: calibration publication failed\n";
                    break;
                }
            }

            if (foxglove_.takeTransformRequest()) {
                if (!publisher_.publishStaticTransforms(sensor_extrinsics_)) {
                    std::cerr << "Runtime: transform publication failed\n";
                    break;
                }
            }

            const bool published = publisher_.publishAvailable(context_.products(),
                                                               [this](const CompletionHandle& completion) {

                        return context_.waitForHost(completion);
                    });

            if (!published) {
                std::cerr << "Runtime: visualization publication failed\n";
                break;
            }

            const auto telemetry_now = std::chrono::steady_clock::now();

            if (foxglove_.runtimeTelemetryChannel().hasSinks() &&
                telemetry_now - last_telemetry_publish_ >=
                std::chrono::seconds(1)) {

                nlohmann::json message;

                message["producers"] = nlohmann::json::array();

                for (const auto& [producer, stats] : producer_execution_stats_) {
                    if (producer == nullptr) continue;

                    message["producers"].push_back({{"name", producer->name()},
                                                    {"considered", stats.considered.load()},
                                                    {"submitted", stats.submitted.load()},
                                                    {"no_work", stats.no_work.load()},
                                                    {"failed", stats.failed.load()},
                                                    {"missing_or_incompatible_input", stats.missing_or_incompatible_input.load()},
                                                    {"rate_limited", stats.rate_limited.load()},
                                                    {"stale_input", stats.stale_input.load()},
                                                    {"superseded", stats.superseded.load()}
                                                });
                }

                const auto& metrics = runtime_metrics();

                message["resources"] = {{"cuda_allocations", metrics.cuda_allocations.load()},
                                        {"cuda_allocated_bytes", metrics.cuda_allocated_bytes.load()},
                                        {"cuda_frees", metrics.cuda_frees.load()},
                                        {"host_to_device_transfers", metrics.host_to_device_transfers.load()},
                                        {"host_to_device_bytes", metrics.host_to_device_bytes.load()},
                                        {"device_to_host_transfers", metrics.device_to_host_transfers.load()},
                                        {"device_to_host_bytes", metrics.device_to_host_bytes.load()},
                                        {"device_to_device_transfers", metrics.device_to_device_transfers.load()},
                                        {"device_to_device_bytes", metrics.device_to_device_bytes.load()},
                                        {"accelerator_waits", metrics.accelerator_waits.load()},
                                        {"host_waits", metrics.host_waits.load()},
                                        {"context_drains", metrics.context_drains.load()}};

                if (!publisher_.publishRuntimeTelemetry(message.dump())) {
                    std::cerr << "Runtime: telemetry publication failed\n";
                    break;
                }

                last_telemetry_publish_ = telemetry_now;
            }

            // processCommands();
            // dispatch(products);
        }
        running_.store(false);
        if (lidar_thread_.joinable()) lidar_thread_.join();
    }

    void Runtime::runLidarSource() {
        /**
        * Resolve the independently clocked LiDAR branch once. LidarScan is a
        * source product, so the expected plan contains only the RPLIDAR producer.
        *
        * Keeping this plan on its own worker prevents blocking SLAMTEC SDK reads
        * from setting the cadence of the camera/stereo branch.
        */
        const auto execution_plan = resolver_.resolve(ProductId::LidarScan);
        if (execution_plan.empty()) {
            std::cerr << "Runtime: no producer available for lidarscan\n";
            return;
        }

        while (running_.load()) {
            for (Producer* producer : execution_plan) {
                if (producer == nullptr) {
                    std::cerr << "Runtime: null producer in lidar execution plan\n";
                    return;
                }

                auto& stats = producer_execution_stats_.at(producer);
                ++stats.considered;

                const SubmitResult result = producer->submit(context_);
                if (result == SubmitResult::Failed) {
                    ++stats.failed;

                    std::cerr << "Runtime: lidar producer failed: " << producer->name() << '\n';

                    return;
                }

                if (result == SubmitResult::NoWork) {
                    ++stats.no_work;
                    continue;
                }

                ++stats.submitted;
            }
        }
    }

    void Runtime::stop() noexcept { running_.store(false); }

    void Runtime::shutdown() {
        stop();

        if (lidar_thread_.joinable()) lidar_thread_.join();


        if (!context_.drain()) {
            std::cerr << "Runtime: failed to drain execution context during shutdown\n";
        }
        publisher_.shutdown();
        foxglove_.shutdown();

        /**
         * Drop published handles before destroying the hardware and processing
         * resources they reference. Producers are destroyed before their backing
         * devices because Graph stores only non-owning producer pointers.
         */
        context_.products().clear();
        producer_execution_state_.clear();
        marker_depth_producer_.reset();
        charuco_pose_producer_.reset();
        depth_producer_.reset();
        stereo_producer_.reset();
        rectification_producer_.reset();
        isp_producer_.reset();
        lidar_producer_.reset();
        camera_producer_.reset();

        pipeline_.shutdown();
        
        if (lidar_) {
            lidar_->shutdown();
            lidar_.reset();
        }
        
        if (camera_) {
            camera_->shutdown();
            camera_.reset();
        }
        
        context_.shutdown();
        initialized_ = false;
    }
}