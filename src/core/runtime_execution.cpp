#include <parallax/core/runtime.hpp>
#include <parallax/core/pipeline.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <parallax/core/history_configuration.hpp>
#include <parallax/core/runtime_metrics.hpp>
#include <parallax/mapping/mapping_metrics.hpp>
#include <parallax/application/foxglove_command.hpp>
#include <parallax/camera/arducam_controls.hpp>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string_view>
#include <limits>
#include <vector>
#include <iostream>

namespace parallax::core {
void Runtime::run(const volatile std::sig_atomic_t& stop_requested) {
        if (!initialized_) return;
        running_.store(true);
        int failed_frames = 0;
        last_telemetry_publish_ = std::chrono::steady_clock::now();

        /**
         * The camera-domain plan follows active demand.
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

        visualization_failed_.store(false);
        visualization_thread_ = std::thread(&Runtime::runVisualization, this);
        if (auto_controller_) auto_control_thread_ = std::thread(&Runtime::runAutoControl, this);
        if (lidar_producer_) lidar_thread_ = std::thread(&Runtime::runLidarSource, this);

        while (running_.load() && !stop_requested) {
            refresh_execution_plan();
            bool frame_failed = false;

            // RequestController owns the persistent requested target/revision.
            // DetectionProducer owns the graph-facing detector query state.
            // The foxglove service handler remains control-plane only:
            // it records intent/demand and never calls NanoOWL directly.
            const auto request_state = request_controller_.state();

            // RequestController owns intent; the producer owns DCF state.
            // Repeating the same revision is intentionally a no-op.
            if (single_target_producer_) {
                if (request_state.tracking_requested) {
                    if (!single_target_producer_->setTarget(request_state.tracking_target, request_state.tracking_query_revision)) {
                        std::cerr << "Runtime: failed to apply tracking target\n";
                        break;
                    }
                }
                else if (!single_target_producer_->targetQuery().empty()) {
                    single_target_producer_->reset();
                }
            }

            const bool tracker_needs_detection = request_state.tracking_requested &&
                                                 single_target_producer_ &&
                                                 single_target_producer_->needsDetection();

            const bool segmentation_conflicts = tracker_needs_detection &&
                                                request_state.segmentation_requested &&
                                                request_state.segmentation_target != request_state.tracking_target;

            /*
            * Tracking borrows the existing detector for acquisition and for
            * bounded semantic refreshes. SAM follows those refresh detections so
            * detector correction, mask, and metric depth retain one observation ID.
            * An explicit segmentation prompt for another target still wins rather
            * than being silently replaced by tracking.
            */
            if (detection_producer_) {
                if (tracker_needs_detection && !segmentation_conflicts) {
                    if (!detection_producer_->setQuery(request_state.tracking_target, request_state.tracking_query_revision)) {
                        std::cerr << "Runtime: failed to apply tracking detection query\n";
                        break;
                    }
                }
                else if (request_state.detection_requested) {
                    if (!detection_producer_->setQuery(request_state.detection_target, request_state.detection_query_revision)) {
                        std::cerr << "Runtime: failed to apply NanoOWL detection query\n";
                        break;
                    }
                }
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

                        // Dynamic demand may activate a producer before a usable generation
                        // exists. That is a scheduling miss, not a runtime failure.        
                        continue;
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

            const auto telemetry_now = std::chrono::steady_clock::now();

            if (foxglove_.runtimeTelemetryChannel().hasSinks() && telemetry_now - last_telemetry_publish_ >= std::chrono::seconds(1)) {

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
                                                    {"superseded", stats.superseded.load()}});
                }

                const auto& metrics = runtime_metrics();

                if (nanoowl_) {
                    const auto detector_metrics = nanoowl_->metrics();
                    message["detection"] = {{"query_revision", detector_metrics.query_revision},
                                            {"query_encoding_count", detector_metrics.query_encoding_count},
                                            {"last_predict_ms", detector_metrics.last_predict_ms}};
                }

                if (single_target_producer_) {
                    const auto& track = single_target_producer_->track();
                    const auto tracker_metrics = single_target_producer_->metrics();

                    message["tracking"] = {{"target", single_target_producer_->targetQuery()},
                                           {"target_revision", single_target_producer_->targetRevision()},
                                           {"tracking", single_target_producer_->tracking()},
                                           {"needs_detection", single_target_producer_->needsDetection()},
                                           {"track_id", track.track_id},
                                           {"lifecycle", static_cast<std::uint8_t>(track.lifecycle)},
                                           {"quality", track.quality},
                                           {"source_sequence", track.source_observation.sequence},
                                           {"detector_sequence", track.last_detector_observation.sequence},
                                           {"tracker_sequence", track.last_tracker_observation.sequence},
                                           {"updates", tracker_metrics.tracker_updates},
                                           {"update_hz", tracker_metrics.tracker_update_hz},
                                           {"skipped_rgb", tracker_metrics.skipped_rgb_observations},
                                           {"sequence_gap_resets", tracker_metrics.sequence_gap_resets},
                                           {"lost_transitions", tracker_metrics.lost_transitions},
                                           {"reacquisition_requests", tracker_metrics.reacquisition_requests},
                                           {"reacquisition_successes", tracker_metrics.reacquisition_successes},
                                           {"detector_refreshes", tracker_metrics.detector_refreshes},
                                           {"detector_refresh_hz", tracker_metrics.detector_refresh_hz},
                                           {"lost_duration_ms", tracker_metrics.lost_duration_ms},
                                           {"resets", tracker_metrics.resets}};
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

                const auto& mm=parallax::mapping::mapping_metrics();
                const auto stage_json=[](const parallax::mapping::StageMetrics& s){ const auto n=s.samples.load(); const auto total=s.total_us.load(); return nlohmann::json{{"samples",n},{"last_ms",static_cast<double>(s.last_us.load())/1000.0},{"mean_ms",n?static_cast<double>(total)/static_cast<double>(n)/1000.0:0.0},{"max_ms",static_cast<double>(s.max_us.load())/1000.0}}; };
                message["mapping"]={{"profiling_sync",mm.profiling_sync.load()},{"depth_integration",stage_json(mm.depth_integration)},{"tsdf_snapshot",stage_json(mm.tsdf_snapshot)},{"color_integration",stage_json(mm.color_integration)},{"mesh_update_and_flatten",stage_json(mm.mesh_update_and_flatten)},{"mesh_snapshot",stage_json(mm.mesh_snapshot)},{"tsdf_publication",stage_json(mm.tsdf_publication)},{"mesh_publication",stage_json(mm.mesh_publication)},{"tsdf_d2h_transfers",mm.tsdf_d2h_transfers.load()},{"tsdf_d2h_bytes",mm.tsdf_d2h_bytes.load()},{"mesh_materialized_host_bytes",mm.mesh_materialized_host_bytes.load()},{"mesh_d2h_bytes_exact",nullptr},{"map_allocated_blocks",mm.map_allocated_blocks.load()},{"map_peak_allocated_blocks",mm.map_peak_allocated_blocks.load()}};

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
        if (auto_control_thread_.joinable()) auto_control_thread_.join();
        if (visualization_thread_.joinable()) visualization_thread_.join();
        
        if (visualization_failed_.load()) {
            std::cerr << "Runtime: visualization worker stopped after a publication error\n";
        }
        if (lidar_thread_.joinable()) lidar_thread_.join();
    }
}
