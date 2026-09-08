#include <parallax/localization/cuvslam_producer.hpp>

#include <parallax/core/execution_context.hpp>

#include <chrono>
#include <exception>
#include <memory>
#include <utility>

namespace parallax::localization {

    namespace {
        std::int64_t timestampNs(
            const parallax::core::ProductMetadata& metadata) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(metadata.timestamp.time_since_epoch()).count();
        }

        LocalizationPose makeLocalizationPose(const CuVslamPoseEstimate& estimate, std::uint64_t epoch) {

            LocalizationPose result{};
            result.timestamp_ns = estimate.timestamp_ns;
            result.epoch = epoch;

            const auto& pose = estimate.world_from_rig->pose;

            result.translation_m = {pose.translation[0],
                                    pose.translation[1],
                                    pose.translation[2]};

            result.rotation_xyzw = {pose.rotation[0],
                                    pose.rotation[1],
                                    pose.rotation[2],
                                    pose.rotation[3]};

            return result;
        }
    }

    CuVslamProducer::CuVslamProducer(CuVslamLocalizer& localizer, parallax::core::ProductStore& store)
                                    : localizer_(localizer), store_(store) {}

    std::string_view CuVslamProducer::name() const noexcept {
        return "localization.cuvslam";
    }

    const std::vector<parallax::core::ProductId>& CuVslamProducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<parallax::core::ProductId>& CuVslamProducer::outputs() const noexcept {
        return outputs_;
    }

    const std::vector<parallax::core::OrderedInputRequirement>& CuVslamProducer::ordered_inputs() const noexcept {
        return ordered_inputs_;
    }

    parallax::core::ExecutionPolicy CuVslamProducer::execution_policy() const noexcept {

        parallax::core::ExecutionPolicy policy{};
        policy.target_hz = 0.0;
        policy.max_input_age_ms = 0.0;
        policy.drop_policy = parallax::core::DropPolicy::Block;
        policy.priority = 10;
        policy.affinity = parallax::core::ResourceAffinity::Gpu;
        policy.stateful = true;

        return policy;
    }

    std::shared_ptr<const CuVslamProducer::GrayProduct> CuVslamProducer::nextInput() const {
        if (last_consumed_) {
            return store_.next_after<parallax::isp::RectifiedStereoGrayFrame>(parallax::core::ProductId::RectifiedGray, *last_consumed_);
        }

        const auto history = store_.history<parallax::isp::RectifiedStereoGrayFrame>(parallax::core::ProductId::RectifiedGray);

        if (history.empty()) return {};

        return history.front();
    }

    void CuVslamProducer::publishState(const parallax::core::ProductMetadata& input_metadata, LocalizationTrackingState tracking) {

        auto metadata = input_metadata;
        metadata.production_timestamp = parallax::core::ExecutionContext::now();
        metadata.valid = true;

        LocalizationState state{};
        state.tracking = tracking;
        state.epoch = epoch_;
        state.consumed_frames = consumed_frames_;
        state.input_gaps = input_gaps_;
        state.session_resets = session_resets_;

        store_.publish(parallax::core::make_product(parallax::core::ProductId::LocalizationState,
                                                    metadata,
                                                    std::make_shared<const LocalizationState>(std::move(state))));
    }

    parallax::core::SubmitResult CuVslamProducer::submit(parallax::core::ExecutionContext& context) {
        if (!localizer_.initialized()) return parallax::core::SubmitResult::Failed;

        const auto input = nextInput();
        if (!input || !input->valid()) return parallax::core::SubmitResult::NoWork;

        const auto observation = input->metadata.observation;
        if (observation.source != parallax::core::SourceId::StereoCamera) {
            return parallax::core::SubmitResult::Failed;
        }

        if (last_consumed_ && observation.sequence > last_consumed_->sequence + 1) {

            ++input_gaps_;

            // If ordered history rolled past us, this is a new local world.
            // Do not pretend the old cuVSLAM session is still continuous.
            if (!localizer_.reset()) return parallax::core::SubmitResult::Failed;

            ++epoch_;
            ++session_resets_;
            last_timestamp_ns_ = -1;
        }

        const std::int64_t timestamp_ns = timestampNs(input->metadata);

        if (last_timestamp_ns_ >= 0 && timestamp_ns <= last_timestamp_ns_) {

            ++input_gaps_;

            if (!localizer_.reset()) return parallax::core::SubmitResult::Failed;

            ++epoch_;
            ++session_resets_;
            last_timestamp_ns_ = -1;
        }

        // cuVSLAM owns its internal CUDA scheduling but does not accept our
        // completion handle, so this is the real host/API readiness boundary.
        if (!context.waitForHost(input->completion)) return parallax::core::SubmitResult::Failed;

        const auto& gray = *input->payload;

        CuVslamFrame frame{};
        frame.left = gray.left.data();
        frame.right = gray.right.data();

        frame.width = static_cast<std::int32_t>(gray.width);
        frame.height = static_cast<std::int32_t>(gray.height);

        frame.left_pitch = static_cast<std::int32_t>(gray.left.pitch());
        frame.right_pitch = static_cast<std::int32_t>(gray.right.pitch());

        frame.timestamp_ns = timestamp_ns;
        frame.gpu_memory = true;

        CuVslamPoseEstimate estimate{};

        try {
            estimate = localizer_.track(frame);
        } catch (const std::exception&) {
            return parallax::core::SubmitResult::Failed;
        }

        last_consumed_ = observation;
        last_timestamp_ns_ = timestamp_ns;
        ++consumed_frames_;

        if (!estimate.valid()) {
            publishState(input->metadata, LocalizationTrackingState::Lost);
            return parallax::core::SubmitResult::Submitted;
        }

        auto metadata = input->metadata;
        metadata.production_timestamp = parallax::core::ExecutionContext::now();
        metadata.valid = true;

        LocalizationOdometry odometry{};
        odometry.pose = makeLocalizationPose(estimate, epoch_);

        store_.publish(parallax::core::make_product(parallax::core::ProductId::LocalizationOdometry,
                                                    metadata,
                                                    std::make_shared<const LocalizationOdometry>(std::move(odometry))));

        publishState(input->metadata, LocalizationTrackingState::Tracking);

        return parallax::core::SubmitResult::Submitted;
    }
}