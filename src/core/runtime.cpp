#include <parallax/core/runtime.hpp>
#include <parallax/core/pipeline.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <parallax/core/history_configuration.hpp>
#include <parallax/core/runtime_metrics.hpp>
#include <parallax/application/foxglove_command.hpp>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <limits>
#include <vector>
#include <iostream>

namespace parallax::core {
    Runtime::~Runtime() { shutdown(); }

    bool Runtime::initialize(const std::filesystem::path& camera_config_path,
                             const std::filesystem::path& sensor_extrinsics_path,
                             const std::filesystem::path& calibration_directory,
                             const std::filesystem::path& nanoowl_engine_path) {

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

        nanoowl_ = std::make_unique<parallax::perception::NanoOwlBridge>();
        if (!nanoowl_->initialize(nanoowl_engine_path)) {
            std::cerr << "Runtime: failed to initialize NanoOWL\n";
            shutdown();
            return false;
        }

        efficientvit_sam_ = std::make_unique<parallax::perception::EfficientVitSam>();

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
        detection_producer_ = std::make_unique<parallax::perception::DetectionProducer>(*nanoowl_, context_.products());

        segmentation_producer_ = std::make_unique<parallax::perception::SegmentationProducer>(
                                                *efficientvit_sam_,
                                                context_.products(),
                                                "models/efficientvit-sam/engines/l0_encoder_fp16.engine",
                                                "models/efficientvit-sam/engines/l0_decoder_fp16.engine");

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
        graph_.register_producer(*detection_producer_);
        graph_.register_producer(*lidar_producer_);
        graph_.register_producer(*segmentation_producer_);

        graph_.finalize();

        configure_product_history(graph_, context_.products());

        /**
         * These are deliberately root products rather than an exhaustive list of
         * intermediates. DependencyResolver derives the required producer subgraph.
         *
         * RectifiedRgb keeps camera -> ISP -> rectification active.
         * Disparity extends that path through stereo matching.
         * MarkerDepth extends it through metric depth and marker pose/depth.
         * LidarScan keeps the independent RPLIDAR source active.
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


        foxglove::ServiceHandler command_handler {
            [this](const foxglove::ServiceRequest& request, foxglove::ServiceResponder&& responder) {
                /**
                 * Foxglove owns request/response transport.
                 *
                 * Runtime remains the composition boundary between transport,
                 * command decoding, and application request ownership.
                 *
                 * This callback performs bounded control-plane work only:
                 * decode -> validate -> mutate request/demand state -> respond.
                 * It never submits producers or waits for perception output.
                 */
                const auto parsed = parallax::application::parse_foxglove_command(request.payloadStr());

                if (!parsed.ok()) {
                    std::move(responder).respondError(parsed.message);
                    return;
                }

                const auto result = request_controller_.apply(parsed.command);
                if (result.applied() && foxglove_.requestStateChannel().hasSinks()) {

                    const auto state = request_controller_.state();

                    nlohmann::json request_state{{"marker_depth_requested", state.marker_depth_requested},
                                                 {"detection_requested", state.detection_requested},
                                                 {"detection_target", state.detection_target},
                                                 {"detection_query_revision", state.detection_query_revision},
                                                 {"tracking_requested", state.tracking_requested},
                                                 {"tracking_target", state.tracking_target},
                                                 {"segmentation_requested", state.segmentation_requested},
                                                 {"segmentation_target", state.segmentation_target}};

                    const std::string serialized_state = request_state.dump();
                    const auto error = foxglove_.requestStateChannel().log(reinterpret_cast<const std::byte*>(
                                                                           serialized_state.data()),
                                                                           serialized_state.size());

                    if (error != foxglove::FoxgloveError::Ok) {
                        std::cerr << "Runtime: request-state publication failed: "
                                  << foxglove::strerror(error) << '\n';
                    }
                }
                nlohmann::json response{{"accepted", true}, {"status",
                                                            result.applied() ? "applied" : "unavailable"},
                                                            {"message", result.message}};

                const std::string serialized = response.dump();

                const auto* data = reinterpret_cast<const std::byte*>(serialized.data());

                std::vector<std::byte> payload{data, data + serialized.size()};
                std::move(responder).respondOk(payload);
            }
        };

        if (!foxglove_.initialize(std::move(foxglove_demand), std::move(command_handler))) {
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
         * The camera-domain plan follows active demand.
         *
         * LiDAR stays on its independent worker below. Everything else is resolved
         * from the same demand accounting used by Application and Foxglove.
         */
        std::uint64_t execution_plan_revision = std::numeric_limits<std::uint64_t>::max();
        std::vector<Producer*> execution_plan;

        const auto refresh_execution_plan = [&]() {
            auto snapshot = resolver_.active_demand();

            if (snapshot.revision == execution_plan_revision) return;

            snapshot.products.erase(std::remove(snapshot.products.begin(),
                                                snapshot.products.end(),
                                                ProductId::LidarScan),
                                                snapshot.products.end());

            execution_plan = resolver_.resolve(snapshot.products);
            execution_plan_revision = snapshot.revision;
        };

        refresh_execution_plan();
        // Pre-create every stats entry before LiDAR starts. The camera thread can
        // change plans later without mutating the unordered_map concurrently.
        for (Producer* producer : graph_.producers()) {
            if (producer != nullptr) {
                producer_execution_stats_.try_emplace(producer);
            }
        }

        lidar_thread_ = std::thread(&Runtime::runLidarSource, this);

        while (running_.load() && !stop_requested) {
            refresh_execution_plan();
            bool frame_failed = false;

            // RequestController owns the persistent requested target/revision.
            // DetectionProducer owns the graph-facing detector query state.
            // The foxglove service handler remains control-plane only:
            // it records intent/demand and never calls NanoOWL directly.
            const auto request_state = request_controller_.state();
            if (request_state.detection_requested && detection_producer_ && !detection_producer_->setQuery(
                                                                            request_state.detection_target,
                                                                            request_state.detection_query_revision)) {

                std::cerr << "Runtime: failed to apply NanoOWL detection query\n";
                break;
            }

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
                    continue;
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
                telemetry_now - last_telemetry_publish_ >= std::chrono::seconds(1)) {

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

                if (nanoowl_) {
                    const auto detector_metrics = nanoowl_->metrics();
                    message["detection"] = {{"query_revision", detector_metrics.query_revision},
                                            {"query_encoding_count", detector_metrics.query_encoding_count},
                                            {"last_predict_ms", detector_metrics.last_predict_ms}};
                }

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

        if (lidar_thread_.joinable()) {
            lidar_thread_.join();
        }

        // Stop physical hardware as soon as its worker can no longer access it.
        if (lidar_) {
            lidar_->shutdown();
        }

        if (!context_.drain()) {
            std::cerr << "Runtime: failed to drain execution context during shutdown\n";
        }

        publisher_.shutdown();
        foxglove_.shutdown();

        request_controller_.reset();

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
        segmentation_producer_.reset();

        if (efficientvit_sam_) {
            efficientvit_sam_->shutdown();
        }
        efficientvit_sam_.reset();

        detection_producer_.reset();

        // Do NOT explicitly shutdown/reset nanoowl_ here.
        // Its lifetime remains owned by Runtime.
        pipeline_.shutdown();
        lidar_.reset();

        if (camera_) {
            camera_->shutdown();
            camera_.reset();
        }

        context_.shutdown();
        initialized_ = false;
    }
}