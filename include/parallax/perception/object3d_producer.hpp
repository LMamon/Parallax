#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/perception/object3d_association.hpp>
#include <parallax/perception/stereo_roi_associator.hpp>

#include <vector>

namespace parallax::perception {

    /**
     * Graph composition point between semantic detection and metric depth.
     *
     * Detection remains independently useful. Object3D demand is what causes
     * the resolver to extend the active graph through Depth and this producer.
     *
     * Temporal/source compatibility is selected here. Calibrated image-space
     * mapping and stereo ROI reduction remain owned by StereoRoiAssociator.
     */
    class Object3DProducer final : public core::Producer {
        public:
            Object3DProducer(StereoRoiAssociator& associator, core::ProductStore& products) noexcept;

            [[nodiscard]] std::string_view name() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<core::ProductId>& outputs() const noexcept override;
            [[nodiscard]] const std::vector<core::CompatibleInputRequirement>& compatible_inputs() const noexcept override;

            [[nodiscard]] core::ExecutionPolicy execution_policy() const noexcept override;

            core::SubmitResult submit(core::ExecutionContext& context) override;

        private:
            static constexpr std::size_t DepthHistoryCapacity = 4;

            StereoRoiAssociator& associator_;
            core::ProductStore& products_;

            Object3DAssociationPolicy policy_{};

            const std::vector<core::ProductId> inputs_{core::ProductId::Detection, core::ProductId::Depth};
            const std::vector<core::ProductId> outputs_{core::ProductId::Object3D};

            const std::vector<core::CompatibleInputRequirement>
                compatible_inputs_{{core::ProductId::Depth, DepthHistoryCapacity}};
    };
}