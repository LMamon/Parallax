#include <parallax/stereo/depth_producer.hpp>

namespace parallax::stereo {
    DepthProducer::DepthProducer(const StereoCalibration& calibration,
                                 parallax::core::ProductStore& store) :
                                 calibration_(calibration),
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
        // The current disparity-to-depth implementation is a CUDA kernel.
        // Accelerator ownership is preserved during the graph refactor rather
        // than replacing working device code with host-side conversion.
        return {parallax::core::ResourceAffinity::Gpu, false};
    }

    parallax::core::SubmitResult DepthProducer::submit() {
        /**
         * current path already allocates the depth output and invokes
         * parallax::cuda::disparityToDepth() on the existing CUDA stream.
         * Repeating that work here would duplicate depth computation rather than
         * migrate its ownership.
         *
         * Once graph execution becomes authoritative, this producer will consume
         * Disparity, retain device residency, invoke the existing CUDA conversion,
         * and publish Depth with the same source timestamp/sequence lineage.
         */
        return parallax::core::SubmitResult::NoWork;
    }
}