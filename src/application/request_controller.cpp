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

                // Phase 11 records valid neural application intent even though the
                // NanoOWL producer is introduced in Phase 12. Demand and execution
                // remain separate concerns.
                state_.detection_requested = true;
                state_.detection_target = command.target;
                state_.detection_query_revision = next_detection_query_revision_++;

                acquire_once(core::ProductId::Detection, detection_demand_owned_);
                return {RequestStatus::Unavailable, "detection requested; producer is not available until Phase 12"};
            }

            case CommandVerb::Track: {
                if (command.target.empty()) {
                    return {RequestStatus::Invalid, "tracking target is empty"};
                }

                // Tracking is persistent application state. Reissuing track with a
                // different target replaces the target without acquiring another
                // Application demand reference.
                state_.tracking_requested = true;
                state_.tracking_target = command.target;

                acquire_once(core::ProductId::Track2D, tracking_demand_owned_);
                return {RequestStatus::Unavailable, "tracking requested; producer is not available yet"};
            }

            case CommandVerb::StopTracking: {
                release_if_owned(core::ProductId::Track2D, tracking_demand_owned_);

                state_.tracking_requested = false;
                state_.tracking_target.clear();
                return {RequestStatus::Applied, "tracking stopped"};
            }
        }
        return {RequestStatus::Invalid, "unsupported command"};
    }

    void RequestController::reset() {
        release_if_owned(core::ProductId::MarkerDepth, marker_depth_demand_owned_);
        release_if_owned(core::ProductId::Detection, detection_demand_owned_);
        release_if_owned(core::ProductId::Track2D, tracking_demand_owned_);

        state_ = {};
    }
}