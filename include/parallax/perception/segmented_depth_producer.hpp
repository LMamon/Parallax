#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/perception/stereo_roi_associator.hpp>

#include <cstddef>
#include <vector>

namespace parallax::perception {

    class SegmentedDepthProducer final : public core::Producer {
        public:
            SegmentedDepthProducer(StereoRoiAssociator& stereo,
                                   core::ProductStore& products) noexcept;

            [[nodiscard]] std::string_view name() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& outputs() const noexcept override;
            [[nodiscard]] const std::vector<core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;
            [[nodiscard]] core::ExecutionPolicy execution_policy() const noexcept override;
            core::SubmitResult submit(core::ExecutionContext& context) override;

        private:
            // Keep this bounded. DepthFrame retains GPU storage and the Nano has
            // already shown that a much larger history can pressure the pipeline.
            static constexpr std::size_t DepthHistoryCapacity = 4;

            StereoRoiAssociator& stereo_;
            core::ProductStore& products_;
            core::SourceObservation last_mask_observation_{};
            std::uint64_t last_query_revision_ = 0;

            const std::vector<core::ProductId> inputs_{
                core::ProductId::Segmentation,
                core::ProductId::Depth
            };
            const std::vector<core::ProductId> outputs_{core::ProductId::SegmentedDepth};
            const std::vector<core::CompatibleInputRequirement> compatible_inputs_{
                {core::ProductId::Depth, DepthHistoryCapacity}
            };
    };
}
