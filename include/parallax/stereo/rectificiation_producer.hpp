#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/stereo/calibration.hpp>
#include <parallax/stereo/rectification.hpp>

#include <vector>

namespace parallax::stereo {
    /**
     * stereoRectifier still owns its own VPI wrappers, remap payloads, warp maps
     * and storage output storage. this producer doesnt change those resources or
     * introduce an additional accelerator copy
     */

     class RectificationProducer final : public parallax::core::Producer {
        public:
            RectificationProducer(StereoRectifier& rectifier, 
                                  const StereoCalibration& calibration,
                                  parallax::core::ProductStore& store);

            [[nodiscord]] std::string_view name() const noexcept override;
            
            [[nodiscord]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscord]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;
        
            parallax::core::SubmitResult submit() override;
            
        private:
            StereoRectifier& rectifier_;
            // immutable startup state for rectification producer
            const StereoCalibration& calibration_;
            parallax::core::ProductStore& store_;

            const std::vector<parallax::core::ProductId> inputs_{
                parallax::core::ProductId::GrayStereo
            };

            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::RectifiedGray
            };
    };
}