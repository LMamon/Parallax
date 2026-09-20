#include <parallax/perception/tracked_object3d_producer.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/object3d.hpp>
#include <parallax/perception/segmentation.hpp>
#include <parallax/tracking/track.hpp>
#include <chrono>
#include <memory>
namespace parallax::perception {
TrackedObject3DProducer::TrackedObject3DProducer(StereoRoiAssociator& stereo,core::ProductStore& products) noexcept:stereo_(stereo),products_(products){}
std::string_view TrackedObject3DProducer::name() const noexcept{return "perception.tracked_object3d";}
const std::vector<core::ProductId>& TrackedObject3DProducer::inputs() const noexcept{return inputs_;}
const std::vector<core::ProductId>& TrackedObject3DProducer::outputs() const noexcept{return outputs_;}
const std::vector<core::CompatibleInputRequirement>& TrackedObject3DProducer::compatible_inputs() const noexcept{return compatible_inputs_;}
core::ExecutionPolicy TrackedObject3DProducer::execution_policy() const noexcept{core::ExecutionPolicy p{};p.drop_policy=core::DropPolicy::Supersede;p.affinity=core::ResourceAffinity::Gpu;return p;}
core::SubmitResult TrackedObject3DProducer::submit(core::ExecutionContext& context){
 const auto track=products_.latest<tracking::Track2D>(core::ProductId::Track2D);
 if(!track||!track->valid()||!track->payload||!track->payload->valid()) return core::SubmitResult::NoWork;

 /*
  * Strong correction path. SAM may finish after DCF has already advanced, so
  * pair the mask only with depth from the mask's source observation and use
  * the detector prompt box stored in the mask. Never attach an old mask to the
  * newest DCF rectangle.
  */
 const auto mask=products_.latest<SegmentationMask>(core::ProductId::Segmentation);
 if(mask&&mask->valid()&&mask->payload&&mask->payload->valid()&&
    mask->metadata.observation!=last_mask_observation_&&
    mask->payload->source_observation==mask->metadata.observation&&
    mask->payload->query==track->payload->target_query&&
    mask->payload->query_revision==track->payload->target_revision&&
    mask->payload->prompt_box.width>0.0F&&mask->payload->prompt_box.height>0.0F){

   const auto mask_depth=products_.find_observation<isp::DepthFrame>(
       core::ProductId::Depth,mask->metadata.observation);

   if(mask_depth&&mask_depth->valid()&&mask_depth->payload){
     auto& lane=context.stereoLane();
     if(!context.waitFor(mask->completion,lane)) return core::SubmitResult::Failed;

     tracking::Track2D semantic_track=*track->payload;
     semantic_track.box=mask->payload->prompt_box;
     semantic_track.quality=mask->payload->confidence;
     semantic_track.source_observation=mask->metadata.observation;
     semantic_track.last_detector_observation=mask->metadata.observation;
     semantic_track.last_tracker_observation=mask->metadata.observation;
     semantic_track.lifecycle=tracking::TrackLifecycle::Tracking;

     Object3D object{};
     if(stereo_.associateTrack(semantic_track,mask->metadata,*mask_depth,context,object)){
       /*
        * associateTrack establishes the exact detector-box metric fallback.
        * refineWithMask then replaces that support with actual SAM membership.
        */
       if(stereo_.refineWithMask(*mask->payload,mask->metadata,*mask_depth,context,object)){
         auto set=std::make_shared<Object3DSet>();
         set->query=track->payload->target_query;
         set->query_revision=track->payload->target_revision;
         set->objects.push_back(std::move(object));
         if(!set->valid()) return core::SubmitResult::Failed;

         auto metadata=mask->metadata;
         metadata.production_timestamp=std::chrono::steady_clock::now();
         products_.publish(core::make_product<Object3DSet>(
             core::ProductId::TrackedObject3D,metadata,std::move(set)));

         last_mask_observation_=mask->metadata.observation;
         return core::SubmitResult::Submitted;
       }
     }

     /*
      * A valid mask with insufficient stereo texture is not fatal. Mark the
      * observation consumed so one weak frame cannot stall the fast DCF path.
      */
     last_mask_observation_=mask->metadata.observation;
   }
 }

 if(track->metadata.observation==last_track_observation_) return core::SubmitResult::NoWork;

 /* Do not spatialize a current DCF box with depth from a neighboring frame. */
 const auto depth=products_.find_observation<isp::DepthFrame>(
     core::ProductId::Depth,track->metadata.observation);
 if(!depth||!depth->valid()||!depth->payload) return core::SubmitResult::NoWork;

 Object3D object{};
 if(!stereo_.associateTrack(*track->payload,track->metadata,*depth,context,object))
   return core::SubmitResult::NoWork;

 auto set=std::make_shared<Object3DSet>();
 set->query=track->payload->target_query;
 set->query_revision=track->payload->target_revision;
 set->objects.push_back(std::move(object));
 if(!set->valid()) return core::SubmitResult::Failed;

 auto metadata=track->metadata;
 metadata.production_timestamp=std::chrono::steady_clock::now();
 products_.publish(core::make_product<Object3DSet>(
     core::ProductId::TrackedObject3D,metadata,std::move(set)));

 last_track_observation_=track->metadata.observation;
 return core::SubmitResult::Submitted;
}
}
