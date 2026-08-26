#include <parallax/stereo/depth_producer.hpp>
#include <parallax/cuda/depth.cuh>
#include <parallax/core/execution_context.hpp>

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

    parallax::core::SubmitResult DepthProducer::submit(parallax::core::ExecutionContext& context) {
        (void)context;
        auto& lane = context.stereoLane();
        const auto disparity = store_.latest<parallax::isp::StereoMatchFrame>(parallax::core::ProductId::Disparity);

        if (!disparity || !disparity->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }
        
        if (!parallax::cuda::disparityToDepth(disparity->payload->disparity, 
                                              depth_.depth, 
                                              static_cast<float>(calibration_.metadata().virtual_fx),
                                              static_cast<float>(calibration_.metadata().baseline_mm / 1000.0),
                                              parallax::isp::StereoMatchFrame::DisparityScale,
                                              lane.cudaHandle())) {

            return parallax::core::SubmitResult::Failed;
        }
        /**
         * Depth remains device-resident when published. Record completion on the
         * CUDA stream that produced it so CPU consumers can wait only when they
         * actually need to observe depth values.
         */
        if (cudaEventRecord(context.depthCompleteEvent(),
                            lane.cudaHandle()) != cudaSuccess) {
            return parallax::core::SubmitResult::Failed;
        }

        auto depth = std::shared_ptr<const parallax::isp::DepthFrame>(&depth_, [](const parallax::isp::DepthFrame*) {});

        store_.publish(parallax::core::make_product(parallax::core::ProductId::Depth,
                                                    disparity->metadata,
                                                    std::move(depth)));

        return parallax::core::SubmitResult::Submitted;
    }
}