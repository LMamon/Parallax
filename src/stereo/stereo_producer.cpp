#include <parallax/stereo/stereo_producer.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/core/execution_context.hpp>

#include <memory>

namespace parallax::stereo {
    StereoProducer::StereoProducer(StereoMatcher& matcher, parallax::core::ProductStore& store) :
                                   matcher_(matcher),
                                   store_(store) {}

    std::string_view StereoProducer::name() const noexcept {
        return "stereo.disparity";
    }

    const std::vector<parallax::core::ProductId>& StereoProducer::inputs() const noexcept {
        return inputs_;
    }
    
    const std::vector<parallax::core::ProductId>& StereoProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy StereoProducer::execution_policy() const noexcept {
        // StereoMatcher already owns the VPI execution path and its backend
        // resources. This phase preserves that specialization instead of moving
        // stereo work onto a generic CUDA/CPU execution path.
        return {parallax::core::ResourceAffinity::Gpu, false};
    }

    parallax::core::SubmitResult StereoProducer::submit(parallax::core::ExecutionContext& context) {
        (void)context;
        auto& lane = context.stereoLane();
        const auto rectified = store_.latest<parallax::isp::RectifiedStereoGrayFrame>(
                                                parallax::core::ProductId::RectifiedGray);
        
        if (!rectified || !rectified->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }

        if (vpiStreamWaitEvent(lane.handle(), context.preprocessCompleteEvent()) != VPI_SUCCESS) {
            return parallax::core::SubmitResult::Failed;
        }
        if (!matcher_.process(lane.handle())) {
            return parallax::core::SubmitResult::Failed;
        }

        if (vpiEventRecord(context.stereoCompleteEvent(), lane.handle()) != VPI_SUCCESS) {
            return parallax::core::SubmitResult::Failed;
        }

        /**
         * StereoMatcher computes disparity and confidence together into one
         * StereoMatchFrame. Both ProductIds intentionally reference that same
         * device-resident payload.
         */
        auto match = std::shared_ptr<const parallax::isp::StereoMatchFrame>(&matcher_.output(),
                                                                            [](const parallax::isp::StereoMatchFrame*) {});

        store_.publish(parallax::core::make_product(parallax::core::ProductId::Disparity,
                                                    rectified->metadata,
                                                    match));

        store_.publish(parallax::core::make_product(parallax::core::ProductId::Confidence,
                                                    rectified->metadata,
                                                    std::move(match)));

        return parallax::core::SubmitResult::Submitted;
    }
}