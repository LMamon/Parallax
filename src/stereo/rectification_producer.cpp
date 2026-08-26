#include <parallax/stereo/rectification_producer.hpp>
#include <parallax/core/execution_context.hpp>
#include <vpi/Event.h>

#include <cuda_runtime.h>
#include <memory>

namespace parallax::stereo {
    RectificationProducer::RectificationProducer(StereoRectifier& rectifier,
                                                 const StereoCalibration& calibration,
                                                 parallax::core::ProductStore& store) :
                                                    rectifier_(rectifier),
                                                    calibration_(calibration),
                                                    store_(store) {}
    
    std::string_view RectificationProducer::name() const noexcept {
        return "stereo.rectification";
    }

    const std::vector<parallax::core::ProductId>& RectificationProducer::inputs() const noexcept {
        return inputs_;
    }
    
    const std::vector<parallax::core::ProductId>& RectificationProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy RectificationProducer::execution_policy() const noexcept {
        /**
         * StereoRectifier currently submits VPI remap work through its existing
         * stream. Resource/stream ownership is intentionally unchanged until the
         * execution-context and VPI-residency phases
         */
        parallax::core::ExecutionPolicy policy{};
        policy.affinity = parallax::core::ResourceAffinity::Gpu;
        policy.stateful = false;
        return policy;
    }

    parallax::core::SubmitResult RectificationProducer::submit(parallax::core::ExecutionContext& context) {
        const auto gray = store_.latest<parallax::isp::StereoGrayFrame>(parallax::core::ProductId::GrayStereo);

        if (!gray || !gray->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }
        auto& lane = context.preprocessLane();

        if (!context.waitFor(gray->completion, lane)) {
            return parallax::core::SubmitResult::Failed;
        }

        /**
         * StereoRectifier was initialized against the same ISP-owned buffers exposed
         * by GrayStereo. No rebinding or device copy is required here.
         */
        if (!rectifier_.process(lane.handle())) {
            return parallax::core::SubmitResult::Failed;
        }

        auto completion = context.recordVpiCompletion(lane.handle());
        if (!completion.valid()) {
            return parallax::core::SubmitResult::Failed;
        }

        auto rectified = std::shared_ptr<const parallax::isp::RectifiedStereoGrayFrame>(&rectifier_.gray(),
                                                                                        [](const parallax::isp::RectifiedStereoGrayFrame*) 
                                                                                        {});

        store_.publish(parallax::core::make_product(parallax::core::ProductId::RectifiedGray,
                                                    gray->metadata,
                                                    std::move(rectified),
                                                    std::move(completion)));

        return parallax::core::SubmitResult::Submitted;
    }

}