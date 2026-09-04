#include <parallax/tracking/single_target_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/image_space.hpp>
#include <parallax/tracking/tracker_ordering.hpp>

#include <algorithm>
#include <chrono>
#include <memory>

namespace parallax::tracking {

    SingleTargetProducer::SingleTargetProducer(core::ProductStore& products) noexcept : products_(products) {}

    std::string_view SingleTargetProducer::name() const noexcept {
        return "single_target_tracking";
    }

    const std::vector<core::ProductId>& SingleTargetProducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<core::ProductId>& SingleTargetProducer::outputs() const noexcept {
        return outputs_;
    }

    core::ExecutionPolicy SingleTargetProducer::execution_policy() const noexcept {
        core::ExecutionPolicy policy{};

        policy.target_hz = 0.0;
        policy.max_input_age_ms = 0.0;
        policy.drop_policy = core::DropPolicy::Supersede;
        policy.priority = 0;
        policy.affinity = core::ResourceAffinity::Gpu;
        policy.stateful = true;

        return policy;
    }

    core::SubmitResult SingleTargetProducer::submit(core::ExecutionContext& context) {
        if (!tracker_.initialized()) {
            return initialize_from_detection(context) ? core::SubmitResult::Submitted : core::SubmitResult::NoWork;
        }
        return update_track(context);
    }

    bool SingleTargetProducer::initialize_from_detection(core::ExecutionContext& context) {
        const auto detection = products_.latest<perception::DetectionSet>(core::ProductId::Detection);

        if (!detection ||
            !detection->valid() ||
            !detection->payload->valid() ||
            detection->payload->empty()) {

            return false;
        }

        if (detection->payload->image_space != perception::ImageSpace::RgbLeft) {
            return false;
        }

        const auto rgb = products_.find_observation<isp::StereoRgbFrame>(core::ProductId::RgbLeft, detection->metadata.observation);

        if (!rgb || !rgb->valid()) return false;

        if (!context.waitFor(rgb->completion, context.neuralCudaLane())) {
            return false;
        }

        const auto best = std::max_element(detection->payload->scores.begin(), detection->payload->scores.end());

        if (best == detection->payload->scores.end()) return false;

        const std::size_t index = static_cast<std::size_t>(std::distance(detection->payload->scores.begin(), best));

        if (!tracker_.initialize(rgb->payload->left, detection->payload->boxes[index])) {
            return false;
        }

        track_ = {};
        track_.track_id = next_track_id_++;
        track_.target_query = detection->payload->query;
        track_.target_revision = detection->payload->query_revision;
        track_.box = detection->payload->boxes[index];
        track_.quality = detection->payload->scores[index];
        track_.lifecycle = TrackLifecycle::Tentative;
        track_.image_space = perception::ImageSpace::RgbLeft;

        track_.source_observation = detection->metadata.observation;
        track_.last_detector_observation = detection->metadata.observation;
        track_.last_tracker_observation = detection->metadata.observation;
        track_.last_detector_timestamp = detection->metadata.timestamp;
        track_.last_tracker_timestamp = detection->metadata.timestamp;

        auto metadata = detection->metadata;
        metadata.production_timestamp = std::chrono::steady_clock::now();

        publish_track(TrackLifecycle::Tentative, track_.box, track_.quality, metadata);

        return true;
    }

    core::SubmitResult SingleTargetProducer::update_track(core::ExecutionContext& context) {

        const auto rgb = products_.latest<isp::StereoRgbFrame>(core::ProductId::RgbLeft);

        if (!rgb || !rgb->valid()) return core::SubmitResult::NoWork;

        const auto ordering = evaluate_tracker_frame(track_.last_tracker_observation, rgb->metadata.observation);

        if (ordering.decision == TrackerFrameDecision::RejectDuplicate ||
            ordering.decision == TrackerFrameDecision::RejectOlder) {

            return core::SubmitResult::NoWork;
        }

        if (ordering.requires_reset()) {
            tracker_.reset();

            auto metadata = rgb->metadata;
            metadata.production_timestamp = std::chrono::steady_clock::now();

            track_.source_observation = rgb->metadata.observation;
            track_.last_tracker_observation = rgb->metadata.observation;
            track_.last_tracker_timestamp = rgb->metadata.timestamp;

            publish_track(TrackLifecycle::Lost, track_.box, 0.0F, metadata);

            return core::SubmitResult::Submitted;
        }

        /*
         * The tracker follows the newest acceptable RGB generation.
         * Detection provenance stays unchanged until a later reacquisition.
         */
        if (!context.waitFor(rgb->completion, context.neuralCudaLane())) {
            return core::SubmitResult::Failed;
        }

        const auto result = tracker_.update(rgb->payload->left);

        auto metadata = rgb->metadata;
        metadata.production_timestamp = std::chrono::steady_clock::now();
        track_.source_observation = rgb->metadata.observation;
        track_.last_tracker_observation = rgb->metadata.observation;
        track_.last_tracker_timestamp = rgb->metadata.timestamp;

        if (!result.tracked) {
            tracker_.reset();
            publish_track(TrackLifecycle::Lost, result.box, result.response, metadata);

            return core::SubmitResult::Submitted;
        }

        publish_track(TrackLifecycle::Tracking, result.box, result.response, metadata);

        return core::SubmitResult::Submitted;
    }

    void SingleTargetProducer::publish_track(TrackLifecycle lifecycle,
                                             const cv::Rect2f& box,
                                             float quality,
                                             const core::ProductMetadata& metadata) {

        track_.lifecycle = lifecycle;
        track_.box = box;
        track_.quality = quality;

        auto published = std::make_shared<Track2D>(track_);

        products_.publish(core::make_product<Track2D>(core::ProductId::Track2D,
                                                      metadata,
                                                      std::move(published),
                                                      core::CompletionHandle::cpu_ready()));
    }

    void SingleTargetProducer::reset() noexcept {
        tracker_.reset();
        track_ = {};
    }
}