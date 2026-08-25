#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>

#include <parallax/isp/isp.hpp>

#include <vector>

namespace parallax::isp {
    /**
     * ISP still owns CUDA streams and allocations used for bayer upload,
     * demosaic, and grayscale conversion. the producer only declares dependency
     * relationship+exposes the existing device-resident outputs
     * 
     * downstream accelerator producers should consume the existing CUDA-backed 
     * frame storage directly.
     */

     class IspProducer final : public parallax::core::Producer {
        public:
            IspProducer(ISP& isp, parallax::core::ProductStore& store);
            [[nodiscard]] std::string_view name() const noexcept override;

            [[nodiscard]] const std::vector<parallax::core::ProductId>& inputs() const noexcept override;
            [[nodiscard]] const std::vector<parallax::core::ProductId>& outputs() const noexcept override;

            [[nodiscard]] parallax::core::ExecutionPolicy execution_policy() const noexcept override;

            parallax::core::SubmitResult submit(parallax::core::ExecutionContext& context) override;
        
        private:
            ISP& isp_;
            parallax::core::ProductStore& store_;

            const std::vector<parallax::core::ProductId> inputs_{
                parallax::core::ProductId::RawStereo
            };
            // using rgbleft for visualization + detection branches. graystereo preserves both ISP grayscale
            // eyes for the rectification/stereo branch
            const std::vector<parallax::core::ProductId> outputs_{
                parallax::core::ProductId::RgbLeft,
                parallax::core::ProductId::GrayStereo,
            };
     };
}