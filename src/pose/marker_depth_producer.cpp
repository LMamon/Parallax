#include <parallax/pose/marker_depth_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

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
        parallax::core::ExecutionPolicy policy{};
        policy.drop_policy = parallax::core::DropPolicy::Supersede;
        policy.affinity = parallax::core::ResourceAffinity::Cpu;
        policy.stateful = false;
        return policy;
    }

    parallax::core::SubmitResult MarkerDepthPoducer::submit(parallax::core::ExecutionContext& context) {
        const auto pose = store_.latest<parallax::pose::CharucoPoseResult>(parallax::core::ProductId::Pose);
        
        if (!pose || !pose->valid()) return parallax::core::SubmitResult::NoWork;
        const auto depth = store_.latest<parallax::isp::DepthFrame>(parallax::core::ProductId::Depth);
        
        if (!depth || !depth->valid()) return parallax::core::SubmitResult::NoWork;

        /*
         * Marker geometry and depth must describe the same camera observation.
         * Marker depth intentionally keeps the stricter exact-observation rule;
         * Object3D's bounded temporal association does not apply here.
         */
        if (!parallax::core::same_source_observation(*depth, pose->metadata.observation)) {
            return parallax::core::SubmitResult::NoWork;
        }

        // Projection remains useful even when stereo cannot provide metric
        // support at the projected marker center.
        auto result = std::make_shared<parallax::pose::CharucoPoseResult>(*pose->payload);

        result->depth_valid = false;
        result->depth_m = 0.0F;

        if (result->pose_valid && result->plane_valid) {
            const int x = static_cast<int>(std::lround(result->projected_center.x));
            const int y = static_cast<int>(std::lround(result->projected_center.y));

            if (x >= 0 &&
                x < static_cast<int>(depth->payload->width) &&
                y >= 0 &&
                y < static_cast<int>(depth->payload->height)) {

                auto& lane = context.stereoLane();
                const cudaStream_t stream = lane.cudaHandle();

                if (stream == nullptr) return parallax::core::SubmitResult::Failed;

                /*
                 * Depth publication can precede completion of its CUDA work.
                 * Chain this reduction behind the exact depth generation rather
                 * than synchronizing the entire stereo pipeline.
                 */
                if (!context.waitFor(depth->completion, lane)) {
                    return parallax::core::SubmitResult::Failed;
                }

                request_host_.center_x = x;
                request_host_.center_y = y;

                /*
                 * Marker depth is a single-ROI use of the same GPU primitive
                 * used by Object3D. Scratch allocations are retained by the
                 * producer and reused across submissions.
                 */
                if (!request_device_.isAllocated() && !request_device_.allocate(1, 1, 1,
                                                                sizeof(parallax::cuda::DepthRoiRequest))) {
                    return parallax::core::SubmitResult::Failed;
                }

                if (!result_device_.isAllocated() && !result_device_.allocate(1, 1, 1,
                                                                sizeof(parallax::cuda::DepthRoiResult))) {
                    return parallax::core::SubmitResult::Failed;
                }

                if (!request_device_.uploadAsync(&request_host_, sizeof(parallax::cuda::DepthRoiRequest), stream)) {
                    return parallax::core::SubmitResult::Failed;
                }

                if (!parallax::cuda::reduceDepthRois(depth->payload->depth,
                                                     request_device_,
                                                     result_device_,
                                                     1,
                                                     RoiRadius,
                                                     stream)) {

                    return parallax::core::SubmitResult::Failed;
                }

                if (!result_device_.downloadAsync(&result_host_, sizeof(parallax::cuda::DepthRoiResult), stream)) {
                    return parallax::core::SubmitResult::Failed;
                }

                // CPU pose metadata needs only the compact ROI result. The full
                // depth frame never crosses the device boundary.
                auto completion = context.recordCudaCompletion(stream);
                if (!completion.valid() || !context.waitForHost(completion)) {
                    return parallax::core::SubmitResult::Failed;
                }

                if (result_host_.valid_samples >= MinValidSamples &&
                    result_host_.sampled_pixels != 0 &&
                    std::isfinite(result_host_.depth_m) && result_host_.depth_m > 0.0F) {

                    result->depth_m = result_host_.depth_m;
                    result->depth_valid = true;
                }
            }
        }

        /*
         * Projection and MarkerDepth share the pose payload type but remain
         * separate graph products because ProductId describes semantics, not
         * merely the underlying C++ type.
         */
        std::shared_ptr<const parallax::pose::CharucoPoseResult> published_result = result;

        store_.publish(parallax::core::make_product(parallax::core::ProductId::Projection,
                                                    pose->metadata,
                                                    published_result));

        store_.publish(parallax::core::make_product(parallax::core::ProductId::MarkerDepth,
                                                    pose->metadata,
                                                    std::move(published_result)));

        return parallax::core::SubmitResult::Submitted;
    }
}