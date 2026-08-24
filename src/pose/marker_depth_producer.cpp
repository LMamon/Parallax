#include <parallax/pose/marker_depth_producer.hpp>

namespace parallax::pose {
    MarkerDepthPoducer::MarkerDepthPoducer(parallax::core::ProductStore& store) : store_(store) {}

    std::string_view MarkerDepthPoducer::name() const noexcept {
        return "pose.marker_depth";
    }

    const std::vector<parallax::core::ProductId>& MarkerDepthPoducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<parallax::core::ProductId>& MarkerDepthPoducer::outputs() const noexcept {
        return outputs_;
    }

    parallax::core::ExecutionPolicy MarkerDepthPoducer::execution_policy() const noexcept {
        return {parallax::core::ResourceAffinity::Cpu, false};
    }

    parallax::core::SubmitResult MarkerDepthPoducer::submit() {
        /**
         * The current compatibility path projects the marker plane during
         * CharucoPose::process(), then samples one CUDA depth pixel at the
         * projected center. Repeating that operation here before graph execution
         * owns the frame would duplicate the existing result.
         *
         * During cutover, Pose and Depth must be retrieved as compatible products,
         * not simply as unrelated latest values. Source sequence/timestamp
         * identity is retained so stale marker geometry cannot be combined with
         * depth from another camera frame.
         *
         * Projection remains a named output because consumers may request marker
         * geometry without requiring metric depth. MarkerDepth is the composed
         * product that introduces stereo demand.
         */

        return parallax::core::SubmitResult::NoWork;
    }
}