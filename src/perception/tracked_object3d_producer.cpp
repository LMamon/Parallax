#include <parallax/perception/tracked_object3d_producer.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/object3d.hpp>
#include <parallax/perception/segmented_depth.hpp>
#include <parallax/tracking/track.hpp>

#include <chrono>
#include <memory>

namespace parallax::perception {
    TrackedObject3DProducer::TrackedObject3DProducer(StereoRoiAssociator& stereo, core::ProductStore& products) noexcept 
                                                                                                    : stereo_(stereo), products_(products) {}

    std::string_view TrackedObject3DProducer::name() const noexcept{ 
        return "perception.tracked_object3d"; 
    }

    const std::vector<core::ProductId>& TrackedObject3DProducer::inputs() const noexcept {
        return inputs_; 
    }
    
    const std::vector<core::ProductId>& TrackedObject3DProducer::outputs() const noexcept {
        return outputs_; 
    }
    
    const std::vector<core::CompatibleInputRequirement>& TrackedObject3DProducer::compatible_inputs() const noexcept {
        return compatible_inputs_; 
    }

    core::ExecutionPolicy TrackedObject3DProducer::execution_policy() const noexcept {
        core::ExecutionPolicy p{};
        p.drop_policy = core::DropPolicy::Supersede;
        p.affinity = core::ResourceAffinity::Gpu;
        return p;
    }

    core::SubmitResult TrackedObject3DProducer::submit(core::ExecutionContext& context) {
        const auto track = products_.latest<tracking::Track2D>(core::ProductId::Track2D);

        if(!track || !track->valid() || !track->payload || !track->payload->valid()) return core::SubmitResult::NoWork;

        /*
        * Strong semantic correction is produced independently by SegmentedDepth.
        * Tracking only attaches persistent identity to that already synchronized
        * mask+depth measurement; it does not own segmentation geometry anymore.
        */
        const auto segmented = products_.latest<SegmentedDepth>(core::ProductId::SegmentedDepth);
        
        if(segmented && segmented->valid() && segmented->payload && segmented->payload->valid() &&
                segmented->metadata.observation != last_mask_observation_ &&
                segmented->payload->query == track->payload->target_query &&
                segmented->payload->query_revision == track->payload->target_revision) {

            Object3D object = segmented->payload->object;
            object.track_id = track->payload->track_id;

            auto set = std::make_shared<Object3DSet>();
            set->query = track->payload->target_query;
            set->query_revision = track->payload->target_revision;
            set->objects.push_back(std::move(object));
            
            if (!set->valid()) return core::SubmitResult::Failed;

            auto metadata = segmented->metadata;
            metadata.production_timestamp = std::chrono::steady_clock::now();
            products_.publish(core::make_product<Object3DSet>(
                core::ProductId::TrackedObject3D, metadata, std::move(set)));

            last_mask_observation_ = segmented->metadata.observation;
            return core::SubmitResult::Submitted;
        }

        if(track->metadata.observation==last_track_observation_) return core::SubmitResult::NoWork;

        /* Do not spatialize a current DCF box with depth from a neighboring frame. */
        const auto depth=products_.find_observation<isp::DepthFrame>(core::ProductId::Depth, track->metadata.observation);

        if(!depth||!depth->valid()||!depth->payload) return core::SubmitResult::NoWork;

        Object3D object{};
        if(!stereo_.associateTrack(*track->payload, track->metadata, *depth, context, object))
        return core::SubmitResult::NoWork;

        auto set = std::make_shared<Object3DSet>();
        set->query = track->payload->target_query;
        set->query_revision = track->payload->target_revision;
        set->objects.push_back(std::move(object));
        
        if (!set->valid()) return core::SubmitResult::Failed;

        auto metadata = track->metadata;
        metadata.production_timestamp = std::chrono::steady_clock::now();
        products_.publish(core::make_product<Object3DSet>(core::ProductId::TrackedObject3D, metadata, std::move(set)));

        last_track_observation_ = track->metadata.observation;
        return core::SubmitResult::Submitted;
    }
}
