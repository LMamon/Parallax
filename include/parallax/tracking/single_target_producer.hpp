#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/tracking/dcf_tracker.hpp>
#include <parallax/tracking/track.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace parallax::tracking {

    class SingleTargetProducer final : public core::Producer {
        public:
            explicit SingleTargetProducer(core::ProductStore& products) noexcept;

            [[nodiscard]] std::string_view name() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& outputs() const noexcept override;
            [[nodiscard]] core::ExecutionPolicy execution_policy() const noexcept override;

            core::SubmitResult submit(core::ExecutionContext& context) override;

            void reset() noexcept;

        private:
            [[nodiscard]] bool initialize_from_detection(core::ExecutionContext& context);

            [[nodiscard]] core::SubmitResult update_track(core::ExecutionContext& context);

            void publish_track(TrackLifecycle lifecycle,
                               const cv::Rect2f& box,
                               float quality,
                               const core::ProductMetadata& metadata);

            core::ProductStore& products_;
            DcfTracker tracker_;

            Track2D track_{};
            std::uint64_t next_track_id_ = 1;

            const std::vector<core::ProductId> inputs_{core::ProductId::RgbLeft, core::ProductId::Detection};
            const std::vector<core::ProductId> outputs_{core::ProductId::Track2D};
    };
}