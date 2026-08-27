#include <parallax/pose/marker_depth_producer.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <parallax/core/execution_context.hpp>

#include <cuda_runtime.h>

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

        if (!pose || !pose->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }

        const auto depth = store_.latest<parallax::isp::DepthFrame>(parallax::core::ProductId::Depth);

        if (!depth || !depth->valid()) {
            return parallax::core::SubmitResult::NoWork;
        }

        /**
         * Marker pose and metric depth must describe the same stereo-camera
         * observation. Freshness alone cannot establish that relationship:
         * a newer depth product may be perfectly fresh while belonging to a
         * different camera frame.
         */
        if (!parallax::core::same_source_observation(*depth, pose->metadata.observation)) {
            return parallax::core::SubmitResult::NoWork;
        }

        /**
         * Keep projection metadata independently available even when no valid metric
         * depth exists at the projected center.
         */
        auto result = std::make_shared<parallax::pose::CharucoPoseResult>(*pose->payload);

        result->depth_valid = false;
        result->depth_m = 0.0f;

        if (result->pose_valid && result->plane_valid) {
            const int x = static_cast<int>(std::lround(result->projected_center.x));
            const int y = static_cast<int>(std::lround(result->projected_center.y));

            if (x >= 0 &&
                x < static_cast<int>(depth->payload->width) &&
                y >= 0 &&
                y < static_cast<int>(depth->payload->height)) {

                const auto* row = reinterpret_cast<const std::uint8_t*>(depth->payload->depth.data()) +
                                                                        static_cast<std::size_t>(y) *
                                                                        depth->payload->depth.pitch();

                const float* pixel = reinterpret_cast<const float*>(row) + x;
                float depth_m = 0.0f;

                /**
                 * SYNCHRONIZATION INVENTORY — REQUIRED CPU OBSERVATION
                 *
                 *
                 * MarkerDepth needs one scalar depth value on the CPU. The Depth product
                 * carries the exact completion generation that produced its device buffer,
                 * so this host wait cannot accidentally synchronize against a newer frame.
                 *
                 * Depth publication itself remains asynchronous.
                 */
                if (!context.waitForHost(depth->completion)) {
                    return parallax::core::SubmitResult::Failed;
                }

                if (cudaMemcpy(&depth_m, pixel, sizeof(float), 
                               cudaMemcpyDeviceToHost) == cudaSuccess &&
                               std::isfinite(depth_m) &&
                               depth_m > 0.0f) {

                    result->depth_m = depth_m;
                    result->depth_valid = true;
                }
            }
        }

        /**
         * Projection and MarkerDepth currently share CharucoPoseResult as their C++
         * payload type but retain separate semantic ProductIds. ProductId identifies 
         * meaning in the graph not just C++ type
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