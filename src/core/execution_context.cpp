#include <parallax/core/execution_context.hpp>

#include <cstdint>

namespace parallax::core {

    ExecutionContext::~ExecutionContext() { shutdown(); }

    bool ExecutionContext::initialize() {
        if (initialized_) { return true; }

        /**
         * Image lanes support CUDA/VIC work. They intentionally do not expose OFA
         * because OFA belongs to the stereo execution domain.
         */
        constexpr std::uint64_t image_backends = VPI_BACKEND_CUDA | VPI_BACKEND_VIC;

        /**
         * Stereo needs VIC for pitch/block-linear conversion and OFA for disparity.
         * CUDA remains available for CUDA work sharing the wrapped stream.
         */
        constexpr std::uint64_t stereo_backends = VPI_BACKEND_CUDA | VPI_BACKEND_VIC | VPI_BACKEND_OFA;

        /**
         * Conventional pose currently contains CPU/OpenCV work, but its execution
         * lane is reserved independently so VPI/CUDA preprocessing can later feed
         * it without serializing stereo.
         */
        constexpr std::uint64_t pose_backends = VPI_BACKEND_CUDA | VPI_BACKEND_VIC;

        if (!image_left_lane_.initialize(image_backends) ||
            !image_right_lane_.initialize(image_backends) ||
            !preprocess_lane_.initialize(image_backends) ||
            !stereo_lane_.initialize(stereo_backends) ||
            !conventional_pose_lane_.initialize(pose_backends)) {

            shutdown();
            return false;
        }

        initialized_ = true;
        return true;
    }

    void ExecutionContext::shutdown() noexcept {
        /**
         * adding explicit draining later once producers actually submit work to
         * these lanes. At this point the context owns the lane lifetimes but the
         * current Pipeline remains the active execution path.
         */
        conventional_pose_lane_.shutdown();
        stereo_lane_.shutdown();
        preprocess_lane_.shutdown();
        image_right_lane_.shutdown();
        image_left_lane_.shutdown();

        initialized_ = false;
    }

} // namespace parallax::core