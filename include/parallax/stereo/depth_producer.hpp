#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>

#include <parallax/stereo/calibration.hpp>

#include <vector>

namespace parallax::stereo {
    /**
     * Depth conversion remains the existing CUDA implementation in
     * parallax::cuda::disparityToDepth(). The producer establishes the semantic
     * dependency between disparity and metric depth without moving the algorithm
     * into the orchestration layer.
     *
     * Stereo calibration is immutable startup geometry rather than a per-frame
     * product. The producer therefore keeps an explicit calibration reference
     * while declaring Disparity as its frame-varying graph input.
     *
     * Pose is intentionally absent from this producer. Metric depth is useful
     * independently of board detection, pose estimation, or visualization.
     */
    class DepthProducer final : public parallax::core::Producer {
        public:
            DepthProducer(const StereoCalibration& calibration, parallax::core::ProductStore& store);

            [[nodiscard]] std::string_view name() const noexcept override;
            
            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;
        
            parallax::core::SubmitResult submit() override;

        private:
            // supplies fx and baseline geometry used by  disparity-to-depth conversion
            // calibration lifetime is owned outside the producer.
            const StereoCalibration& calibration_;

            parallax::core::ProductStore& store_;
            const std::vector<parallax::core::ProductId> inputs_{
                parallax::core::ProductId::Disparity
            };

            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::Depth
            };
    };
}