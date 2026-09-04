#pragma once

#include <parallax/core/dependency_resolver.hpp>
#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/tracking/dcf_tracker.hpp>
#include <parallax/tracking/track.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace parallax::tracking {

    struct SingleTargetMetrics {
        std::uint64_t tracker_updates = 0;
        std::uint64_t skipped_rgb_observations = 0;
        std::uint64_t sequence_gap_resets = 0;
        std::uint64_t lost_transitions = 0;
        std::uint64_t reacquisition_requests = 0;
        std::uint64_t reacquisition_successes = 0;
        std::uint64_t detector_refreshes = 0;
        std::uint64_t resets = 0;

        double tracker_update_hz = 0.0;
        double detector_refresh_hz = 0.0;
        double lost_duration_ms = 0.0;
    };

    class SingleTargetProducer final : public core::Producer {
        public:
            SingleTargetProducer(core::ProductStore& products, core::DependencyResolver& resolver) noexcept;

            [[nodiscard]] std::string_view name() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] const std::vector<core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;

            [[nodiscard]] core::ExecutionPolicy execution_policy() const noexcept override;

            [[nodiscard]] bool setTarget(const std::string& target, std::uint64_t revision);

            [[nodiscard]] std::string_view targetQuery() const noexcept { return target_query_; }
            [[nodiscard]] std::uint64_t targetRevision() const noexcept { return target_revision_; }
            [[nodiscard]] bool needsDetection() const noexcept { return reacquisition_needed_; }
            [[nodiscard]] const Track2D& track() const noexcept { return track_; }
            [[nodiscard]] bool tracking() const noexcept { return tracker_.initialized(); }
            [[nodiscard]] SingleTargetMetrics metrics() const noexcept;

            core::SubmitResult submit(core::ExecutionContext& context) override;

            void reset() noexcept;

        private:
            [[nodiscard]] bool initialize_from_detection(core::ExecutionContext& context);
            [[nodiscard]] core::SubmitResult update_track(core::ExecutionContext& context);

            void begin_reacquisition();
            void clear_reacquisition() noexcept;

            void publish_track(TrackLifecycle lifecycle,
                               const cv::Rect2f& box,
                               float quality,
                               const core::ProductMetadata& metadata);

            core::ProductStore& products_;
            core::DependencyResolver& resolver_;

            DcfTracker tracker_;
            Track2D track_{};

            std::string target_query_;
            std::uint64_t target_revision_ = 0;
            std::uint64_t next_track_id_ = 1;

            bool reacquisition_needed_ = false;
            bool detection_demand_owned_ = false;
            std::chrono::steady_clock::time_point reacquisition_started_at_{};

            const std::vector<core::ProductId> inputs_{core::ProductId::RgbLeft};
            const std::vector<core::ProductId> outputs_{core::ProductId::Track2D};
            SingleTargetMetrics metrics_{};

            std::chrono::steady_clock::time_point first_tracker_update_{};
            std::chrono::steady_clock::time_point first_detector_refresh_{};
            std::chrono::steady_clock::time_point lost_since_{};
            
            // Keep only enough RGB history to recover the image used by a detector result.
            const std::vector<core::CompatibleInputRequirement> compatible_inputs_{{core::ProductId::RgbLeft, 2}};
    };
}