#include <parallax/stereo/depth_producer.hpp>
#include <parallax/cuda/depth.cuh>

#include <memory>

namespace parallax::stereo {
    DepthProducer::DepthProducer(const StereoCalibration& calibration,
                                 parallax::isp::DepthFrame& depth,
                                 parallax::vpi::Stream& stream,
                                 parallax::core::ProductStore& store) :
                                 calibration_(calibration),
                                 depth_(depth),
                                 stream_(stream),
                                 store_(store) {}

    std::string_view DepthProducer::name() const noexcept {
        return "stereo.depth";
    }

    const std::vector<parallax::core::ProductId>& DepthProducer::inputs() const noexcept {
        return inputs_;
    }
    
    const std::vector<parallax::core::ProductId>& DepthProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy DepthProducer::execution_policy() const noexcept {
        return {parallax::core::ResourceAffinity::Gpu, false};
    }

    parallax::core::SubmitResult DepthProducer::submit() {
        const auto disparity = store_.latest<parallax::isp::StereoMatchFrame>(parallax::core::ProductId::Disparity);

        if (!disparity || !disparity->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }
        
        if (!parallax::cuda::disparityToDepth(disparity->payload->disparity, 
                                              depth_.depth, 
                                              static_cast<float>(calibration_.metadata().virtual_fx),
                                              static_cast<float>(calibration_.metadata().baseline_mm / 1000.0),
                                              parallax::isp::StereoMatchFrame::DisparityScale,
                                              stream_.cudaHandle())) {

            return parallax::core::SubmitResult::Failed;
        }
        /**
         * Preserve the existing synchronization boundary before CPU pose/depth
         * association can inspect results from this stream.
         */
        if (!stream_.synchronize()) {
            return parallax::core::SubmitResult::Failed;
        }

        auto depth = std::shared_ptr<const parallax::isp::DepthFrame>(&depth_, [](const parallax::isp::DepthFrame*) {});

        store_.publish(parallax::core::make_product(parallax::core::ProductId::Depth,
                                                    disparity->metadata,
                                                    std::move(depth)));

        return parallax::core::SubmitResult::Submitted;
    }
}