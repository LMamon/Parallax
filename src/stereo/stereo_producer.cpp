#include <parallax/stereo/stereo_producer.hpp>

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

    parallax::core::SubmitResult StereoProducer::submit() {
        /**
         * the compatibility path already submits StereoMatcher::process() after
         * rectification. Running it here now would duplicate disparity work and
         * interfere with the existing VPI resource lifecycle.
         *
         * Once graph execution becomes authoritative, this producer consumes th
         * completed RectifiedGray product, invokes the existing matcher, and
         * publishes handles to its disparity/confidence storage. No host staging
         * belongs at this graph boundary.
         */
        return parallax::core::SubmitResult::NoWork;
    }
}