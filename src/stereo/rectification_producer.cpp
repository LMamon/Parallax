#include <parallax/stereo/rectificiation_producer.hpp>

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
        return {parallax::core::ResourceAffinity::Gpu, false};
    }

    parallax::core::SubmitResult RectificationProducer::submit() {
        /**
         * The active compatibility path already calls StereoRectifier::process()
         * against ISP-owned device buffers. Submitting here as well would run the
         * same remap twice and would not represent graph execution correctly.
         *
         * At runtime cutover, this producer will consume the compatible ISP gray
         * product, call the existing StereoRectifier unchanged, and publish its
         * RectifiedStereoGrayFrame without a host round trip.
         */
         
        return parallax::core::SubmitResult::NoWork;
    }

}