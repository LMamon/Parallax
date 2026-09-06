#include <parallax/perception/object3d_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/object3d.hpp>
#include <parallax/perception/segmentation.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>

namespace parallax::perception {
    Object3DProducer::Object3DProducer(StereoRoiAssociator& associator, core::ProductStore& products) noexcept
                                                                    : associator_(associator), products_(products) {}

    std::string_view Object3DProducer::name() const noexcept {
        return "perception.object3d";
    }

    const std::vector<core::ProductId>& Object3DProducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<core::ProductId>& Object3DProducer::outputs() const noexcept {
        return outputs_;
    }

    const std::vector<core::CompatibleInputRequirement>& Object3DProducer::compatible_inputs() const noexcept {
        return compatible_inputs_;
    }

    core::ExecutionPolicy Object3DProducer::execution_policy() const noexcept {
        core::ExecutionPolicy policy{};
        
        // StereoRoiAssociator launches bounded CUDA work and performs only the
        // compact host observation needed to publish CPU Object3D metadata.
        policy.drop_policy = core::DropPolicy::Supersede;
        policy.affinity = core::ResourceAffinity::Gpu;
        policy.stateful = false;

        return policy;
    }

    core::SubmitResult Object3DProducer::submit(core::ExecutionContext& context) {
        const auto detections = products_.latest<DetectionSet>(core::ProductId::Detection);
        if (!detections || !detections->valid() || !detections->payload) {
            return core::SubmitResult::NoWork;
        }

        // Metric selection is exact-observation first, then bounded nearest
        // same-source history. "Latest Depth" by itself is never sufficient
        // evidence that the frame belongs to this semantic observation.
        const auto depth_match = find_metric_observation<isp::DepthFrame>(products_,
                                                                          core::ProductId::Depth,
                                                                          detections->metadata,
                                                                          policy_);

        if (!depth_match.matched() || !depth_match.product) {
            return core::SubmitResult::NoWork;
        }

        auto objects = std::make_shared<Object3DSet>();
        if (!associator_.associate(*detections->payload,
                                   detections->metadata,
                                   *depth_match.product,
                                   context,
                                   *objects)) {

            return core::SubmitResult::Failed;
        }

        /*
        * Segmentation is optional enrichment. Do not make it a graph dependency:
        * Object3D demand alone must never activate EfficientViT-SAM.
        */
        const auto segmentation = products_.latest<SegmentationMask>( core::ProductId::Segmentation);

        if (segmentation &&
            segmentation->valid() &&
            segmentation->payload &&
            segmentation->payload->valid() &&
            segmentation->metadata.observation == detections->metadata.observation &&
            segmentation->payload->query_revision == detections->payload->query_revision &&
            segmentation->payload->source_observation == detections->metadata.observation) {

            /*
            * Preserve the segments highest-confidence detection 
            * exact selection policy here rather than guessing which
            * Object3D the mask belongs to.
            */
            std::size_t selected = 0;

            if (!detections->payload->empty()) {
                for (std::size_t i = 1; i < detections->payload->scores.size(); ++i) {

                    if (detections->payload->scores[i] > detections->payload->scores[selected]) {
                        selected = i;
                    }
                }

                auto object_it = std::find_if(objects->objects.begin(),
                                              objects->objects.end(),
                                              [selected](const Object3D& object) {

                            return object.semantic_index == selected;
                        });

                if (object_it != objects->objects.end()) {
                    auto& lane = context.stereoLane();

                    if (!context.waitFor(segmentation->completion, lane)) return core::SubmitResult::Failed;

                    /*
                    * Failure to obtain enough stereo support under the mask is not an
                    * Object3D failure. Keep the already-valid StereoRoi result.
                    */
                    (void)associator_.refineWithMask(*segmentation->payload, segmentation->metadata, *depth_match.product, context, *object_it);
                }
            }
        }

        /*
         * A valid detection query may legitimately have no supported 3D
         * objects. Publish that empty Object3DSet so downstream consumers can
         * distinguish "association ran and found none" from "producer absent."
         */
        if (!objects->valid()) return core::SubmitResult::Failed;

        /*
         * Product-level provenance remains the semantic observation being
         * enriched. Each Object3D independently preserves the selected metric
         * observation and source delta.
         */
        auto metadata = detections->metadata;
        metadata.production_timestamp = std::chrono::steady_clock::now();

        products_.publish(core::make_product<Object3DSet>(core::ProductId::Object3D,
                                                          metadata,
                                                          std::move(objects),
                                                          core::CompletionHandle::cpu_ready()));

        return core::SubmitResult::Submitted;
    }
}