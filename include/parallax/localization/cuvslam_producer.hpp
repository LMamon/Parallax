#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/localization/cuvslam_localizer.hpp>
#include <parallax/localization/localization.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace parallax::localization {

    class CuVslamProducer final : public parallax::core::Producer {
        public:
            static constexpr std::size_t InputHistoryCapacity = 8;
            static constexpr std::size_t TrajectoryCapacity = 4096;

            CuVslamProducer(CuVslamLocalizer& localizer, parallax::core::ProductStore& store);

            [[nodiscard]] std::string_view name() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::OrderedInputRequirement>& ordered_inputs() const noexcept override;
            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;

            parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;

            [[nodiscard]] std::optional<parallax::core::SourceObservation> last_consumed() const noexcept { return last_consumed_; }
            [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

        private:
            using GrayProduct = parallax::core::Product<parallax::isp::RectifiedStereoGrayFrame>;

            [[nodiscard]] std::shared_ptr<const GrayProduct> nextInput() const;

            void publishState(const parallax::core::ProductMetadata& metadata, LocalizationTrackingState tracking);
            void publishTrajectory(const parallax::core::ProductMetadata& metadata, const LocalizationPose& pose);

            CuVslamLocalizer& localizer_;
            parallax::core::ProductStore& store_;

            // Localization is the exception to the normal latest-frame path.
            // This cursor keeps cuVSLAM moving through retained camera generations in order.
            std::optional<parallax::core::SourceObservation> last_consumed_;
            std::vector<LocalizationPose> trajectory_;

            std::int64_t last_timestamp_ns_ = -1;

            std::uint64_t epoch_ = 0;
            std::uint64_t consumed_frames_ = 0;
            std::uint64_t input_gaps_ = 0;
            std::uint64_t session_resets_ = 0;

            const std::vector<parallax::core::ProductId> inputs_{parallax::core::ProductId::RectifiedGray};

            const std::vector<parallax::core::ProductId> outputs_{parallax::core::ProductId::LocalizationOdometry,
                                                                  parallax::core::ProductId::LocalizationPose,
                                                                  parallax::core::ProductId::LocalizationTrajectory,
                                                                  parallax::core::ProductId::LocalizationState};

            const std::vector<parallax::core::OrderedInputRequirement> ordered_inputs_{{parallax::core::ProductId::RectifiedGray, 
                                                                                        InputHistoryCapacity}};
    };
}