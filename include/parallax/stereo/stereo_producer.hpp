#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>

#include <parallax/stereo/matcher.hpp>

#include <vector>

namespace parallax::stereo {
    /**
     * StereoMatcher remains responsible for VPI stereo-disparity resources,
     * backend-specific image representations, matcher parameters, and output
     * storage. This producer only describes the dependency boundary around that
     * verified implementation.
     *
     * Disparity and confidence are separate graph products even though the
     * matcher computes them together. Consumers can therefore request either
     * semantic result without coupling themselves to StereoMatcher internals.
     */

     class StereoProducer final : public parallax::core::Producer {
        public: 
            StereoProducer(StereoMatcher& matcher, parallax::core::ProductStore& store);

            [[nodiscard]] std::string_view name() const noexcept override;
            
            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;
        
            parallax::core::SubmitResult submit() override;

        private:
            StereoMatcher& matcher_;
            parallax::core::ProductStore& store_;

            // Stereo matching operates only on calibrated + rectified grayscale input
            // RGB remains on its independent graph branch
            const std::vector<parallax::core::ProductId> inputs_{
                parallax::core::ProductId::RectifiedGray
            };

            // matcher produces both outputs from one VPI stereo submission
            // but remain independently addressable products
            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::Disparity,
                parallax::core::ProductId::Confidence
            };
     };
}