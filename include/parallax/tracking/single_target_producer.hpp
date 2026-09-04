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

            // Keep only enough RGB history to recover the image used by a detector result.
            const std::vector<core::CompatibleInputRequirement> compatible_inputs_{{core::ProductId::RgbLeft, 2}};
    };
}