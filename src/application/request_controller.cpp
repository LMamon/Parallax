#include <parallax/application/request_controller.hpp>

#include <utility>

namespace parallax::application {

    RequestController::RequestController(core::DependencyResolver& resolver) noexcept : resolver_(resolver) {}

    void RequestController::acquire_once(core::ProductId product, bool& owned) {
        if (owned) return;

        resolver_.acquire(product, core::DemandSource::Application);
        owned = true;
    }

    void RequestController::release_if_owned(core::ProductId product, bool& owned) {
        if (!owned) return;

        resolver_.release(product, core::DemandSource::Application);
        owned = false;
    }

    RequestResult RequestController::apply(const Command& command) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        switch (command.verb) {
            case CommandVerb::MarkerDepth: {
                acquire_once(core::ProductId::MarkerDepth, marker_depth_demand_owned_);

                state_.marker_depth_requested = true;
                return {RequestStatus::Applied, "marker depth requested"};
            }

            case CommandVerb::Detect: {
                if (command.target.empty()) {
                    return {RequestStatus::Invalid, "detection target is empty"};
                }

                /**
                 * Detection commands establish persistent application intent and
                 * Detection demand.
                 *
                 * Query identity advances independently from camera generations so a
                 * late result from a replaced target cannot be published as though it
                 * belongs to the current request.
                 *
                 * Runtime observes this state at the execution boundary and applies
                 * the query to DetectionProducer. RequestController never calls the
                 * detector directly.
                 */
                state_.detection_requested = true;
                state_.detection_target = command.target;
                state_.detection_query_revision = next_detection_query_revision_++;

                state_.detection_depth_requested = command.depth == DepthRequest::Yes;

                acquire_once(core::ProductId::Detection, detection_demand_owned_);

                if (state_.detection_depth_requested) {
                    acquire_once(core::ProductId::Object3D, object3d_demand_owned_);
                } else {
                    /*
                    * Replacing "detect cup depth=yes" with ordinary detection or
                    * depth=no must remove only this controller's Object3D demand.
                    */
                    release_if_owned(core::ProductId::Object3D, object3d_demand_owned_);
                }

                return {RequestStatus::Applied,
                        state_.detection_depth_requested
                        ? "detection with depth requested"
                        : "detection requested"};
            }   

            case CommandVerb::Track: {
                if (command.target.empty()) {
                    return {RequestStatus::Invalid, "tracking target is empty"};
                }

                /*
                * Repeating the same target preserves its lifecycle. A replacement gets a
                * new revision so Runtime can deterministically reset the tracker.
                */
                if (!state_.tracking_requested || state_.tracking_target != command.target) {
                    state_.tracking_query_revision = next_tracking_query_revision_++;
                }

                state_.tracking_requested = true;
                state_.tracking_target = command.target;

                acquire_once(core::ProductId::Track2D, tracking_demand_owned_);
                return {RequestStatus::Applied, "tracking requested"};
            }

            case CommandVerb::Segment: {
                if (command.target.empty()) {
                    return {RequestStatus::Invalid, "segmentation target is empty"};
                }

                /**
                 * Segmentation is explicitly requested downstream work.
                 *
                 * It also requires Detection because the selected detection supplies
                 * the box prompt. Ordinary Detect commands never acquire Segmentation
                 * demand.
                 *
                 * Detection query revision remains the canonical prompt identity used
                 * to reject masks produced from a replaced target.
                 */
                state_.detection_requested = true;
                state_.detection_target = command.target;
                state_.detection_query_revision = next_detection_query_revision_++;
                state_.detection_depth_requested = false;

                release_if_owned(core::ProductId::Object3D, object3d_demand_owned_);

                state_.segmentation_requested = true;
                state_.segmentation_target = command.target;

                acquire_once(core::ProductId::Detection, detection_demand_owned_);
                acquire_once(core::ProductId::Segmentation, segmentation_demand_owned_);

                return {RequestStatus::Applied, "segmentation requested"};
            }

            case CommandVerb::StopTracking: {
                release_if_owned(core::ProductId::Track2D, tracking_demand_owned_);

                state_.tracking_requested = false;
                state_.tracking_target.clear();
                state_.tracking_query_revision = 0;

                return {RequestStatus::Applied, "tracking stopped"};
            }
        }
        return {RequestStatus::Invalid, "unsupported command"};
    }

    void RequestController::reset() {
        std::lock_guard<std::mutex> lock(state_mutex_);
        release_if_owned(core::ProductId::MarkerDepth, marker_depth_demand_owned_);
        release_if_owned(core::ProductId::Detection, detection_demand_owned_);
        release_if_owned(core::ProductId::Segmentation, segmentation_demand_owned_);
        release_if_owned(core::ProductId::Track2D, tracking_demand_owned_);
        release_if_owned(core::ProductId::Object3D, object3d_demand_owned_);

        state_ = {};
    }

    RequestState RequestController::state() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

}