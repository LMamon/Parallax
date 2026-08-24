#include <parallax/pose/charuco_pose_producer.hpp>

namespace parallax::pose {
    CharucoPoseProducer::CharucoPoseProducer(CharucoPose& pose,
                                             const parallax::stereo::StereoCalibration& calibration,
                                             parallax::core::ProductStore& store) :
                                                pose_(pose),
                                                calibration_(calibration),
                                                store_(store) {}

    std::string_view CharucoPoseProducer::name() const noexcept {
        return "pose.charuco";
    }

    const std::vector<parallax::core::ProductId>& CharucoPoseProducer::inputs() const noexcept {
        return inputs_;
    }
    
    const std::vector<parallax::core::ProductId>& CharucoPoseProducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy CharucoPoseProducer::execution_policy() const noexcept{
        // the current charuco path uses openCV board detection and pose estimation on host-visible grayscale.
        // keeping that execution characteristic explicit instead of pretending this is a GPU producer
        return {parallax::core::ResourceAffinity::Cpu, false};
    }

    parallax::core::SubmitResult CharucoPoseProducer::submit() {
        /**
         * The compatibility path already invokes CharucoPose::process() for the
         * current rectified frame. Calling it here now would duplicate detection
         * and pose estimation rather than transfer orchestration ownership.
         *
         * At graph cutover this producer will consume the compatible
         * RectifiedGray product, run the existing CharucoPose implementation,
         * and publish compact pose metadata with the source frame identity and
         * timestamp preserved.
         *
         * Projection and marker-depth composition stay outside this producer.
         * Those are derived products with different dependencies and are exposed
         * separately in the next migration commit.
         */

        return parallax::core::SubmitResult::NoWork;
    }
}