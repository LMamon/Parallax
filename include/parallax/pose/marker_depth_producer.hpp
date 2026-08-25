#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>

#include <vector>

namespace parallax::pose {
    /**
     * ChArUco detection and pose estimation remain owned by CharucoPoseProducer.
     * Stereo depth remains independently useful through DepthProducer. This
     * producer represents the point where the otherwise independent branches
     * meet.
     *
     * The compatibility rule is part of the product contract: marker geometry
     * must never be associated with an unrelated depth frame merely because that
     * depth happens to be the newest value in ProductStore.
     */
    class MarkerDepthPoducer final : public parallax::core::Producer {
        public:
            explicit MarkerDepthPoducer(parallax::core::ProductStore& store);

            [[nodiscard]] std::string_view name() const noexcept override;
            
            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;
        
            parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;
        
        private:
            parallax::core::ProductStore& store_;
            // marker depth is a composition point between independent pose and stereo barnches
            // requesteing pose alone doesnt activate stereo
            const std::vector<parallax::core::ProductId> inputs_{
                parallax::core::ProductId::Pose,
                parallax::core::ProductId::Depth
            };

            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::Projection,
                parallax::core::ProductId::MarkerDepth
            };
    };
}