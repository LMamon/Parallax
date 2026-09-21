#include <parallax/perception/segmented_depth_producer.hpp>

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/segmentation.hpp>
#include <parallax/perception/segmented_depth.hpp>

#include <chrono>
#include <memory>

namespace parallax::perception {

    SegmentedDepthProducer::SegmentedDepthProducer(StereoRoiAssociator& stereo,
                                                   core::ProductStore& products) noexcept
        : stereo_(stereo), products_(products) {}

    std::string_view SegmentedDepthProducer::name() const noexcept {
        return "perception.segmented_depth";
    }

    const std::vector<core::ProductId>& SegmentedDepthProducer::inputs() const noexcept {
        return inputs_;
    }

    const std::vector<core::ProductId>& SegmentedDepthProducer::outputs() const noexcept {
        return outputs_;
    }

    const std::vector<core::CompatibleInputRequirement>& SegmentedDepthProducer::compatible_inputs() const noexcept {
        return compatible_inputs_;
    }

    core::ExecutionPolicy SegmentedDepthProducer::execution_policy() const noexcept {
        core::ExecutionPolicy policy{};
        policy.drop_policy = core::DropPolicy::Supersede;
        policy.affinity = core::ResourceAffinity::Gpu;
        policy.stateful = false;
        return policy;
    }

    core::SubmitResult SegmentedDepthProducer::submit(core::ExecutionContext& context) {
        const auto mask = products_.latest<SegmentationMask>(core::ProductId::Segmentation);
        if (!mask || !mask->valid() || !mask->payload || !mask->payload->valid()) {
            return core::SubmitResult::NoWork;
        }

        if (mask->metadata.observation == last_mask_observation_ &&
            mask->payload->query_revision == last_query_revision_) {
            return core::SubmitResult::NoWork;
        }

        /*
         * Spatial segmentation is an exact-observation product. A neighboring
         * depth frame may look plausible but would make the semantic boundary
         * and metric surface describe different moments.
         */
        const auto depth = products_.find_observation<isp::DepthFrame>(
            core::ProductId::Depth, mask->metadata.observation);
        if (!depth || !depth->valid() || !depth->payload) {
            return core::SubmitResult::NoWork;
        }

        auto& lane = context.stereoLane();
        if (!context.waitFor(mask->completion, lane)) {
            return core::SubmitResult::Failed;
        }

        Object3D object{};
        if (!stereo_.associateMask(*mask->payload, mask->metadata, *depth, context, object)) {
            // Do not consume the mask: retained exact depth may become available
            // on a later graph pass while the bounded history still owns it.
            return core::SubmitResult::NoWork;
        }

        auto segmented = std::make_shared<SegmentedDepth>();
        segmented->query = mask->payload->query;
        segmented->query_revision = mask->payload->query_revision;
        segmented->source_observation = mask->metadata.observation;

        segmented->object = std::move(object);
        if (!segmented->valid()) return core::SubmitResult::Failed;

        auto metadata = mask->metadata;
        metadata.production_timestamp = std::chrono::steady_clock::now();
        
        products_.publish(core::make_product<SegmentedDepth>(core::ProductId::SegmentedDepth, metadata, std::move(segmented)));

        last_mask_observation_ = mask->metadata.observation;
        last_query_revision_ = mask->payload->query_revision;
        return core::SubmitResult::Submitted;
    }
}
