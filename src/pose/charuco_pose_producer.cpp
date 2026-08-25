#include <parallax/pose/charuco_pose_producer.hpp>
#include <parallax/isp/frame_types.hpp>

#include <memory>

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

    parallax::core::SubmitResult CharucoPoseProducer::submit(parallax::core::ExecutionContext& context) {
        (void)context;
        const auto rectified = store_.latest<parallax::isp::RectifiedStereoGrayFrame>(
                                                parallax::core::ProductId::RectifiedGray);

        if (!rectified || !rectified->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }

        auto result = std::make_shared<parallax::pose::CharucoPoseResult>();

        if (!pose_.process(*rectified->payload, *result)) {
            return parallax::core::SubmitResult::Failed;
        }

        // Metric stereo depth is composed later. Pose stays independently usable.
        result->depth_valid = false;
        result->depth_m = 0.0f;

        // Template deduction doesnt perform const conversion. so finish 
        // mutating the result, then explicitly convert it to a const shared pointer before publication
        std::shared_ptr<const parallax::pose::CharucoPoseResult> published_result = std::move(result);

        store_.publish(parallax::core::make_product(parallax::core::ProductId::Pose,
                                                    rectified->metadata,
                                                    std::move(published_result)));

        return parallax::core::SubmitResult::Submitted;
    }
}