#pragma once

#include <parallax/application/command.hpp>
#include <parallax/core/dependency_resolver.hpp>
#include <parallax/core/product_id.hpp>

#include <string>
#include <cstdint>
#include <mutex>

namespace parallax::application {

    enum class RequestStatus {
        Applied, Unavailable, Invalid
    };

    struct RequestResult {
        RequestStatus status = RequestStatus::Invalid;
        std::string message;

        [[nodiscard]] bool applied() const noexcept {
            return status == RequestStatus::Applied;
        }
    };

    // Persistent application-level state derived from commands.
    //
    // This is intentionally separate from Command. A Command is a transient
    // requested transition; RequestState describes application intent that
    // remains true after that command has been handled.
    struct RequestState {
        bool marker_depth_requested = false;

        bool detection_requested = false;
        std::string detection_target;
        std::uint64_t detection_query_revision = 0;

        bool segmentation_requested = false;
        std::string segmentation_target;

        bool tracking_requested = false;
        std::string tracking_target;
            
        std::uint64_t tracking_query_revision = 0;
    };

    class RequestController {
        public:
            explicit RequestController(core::DependencyResolver& resolver) noexcept;

            // Translate application intent into graph demand/state.
            //
            // This function never executes producers. DependencyResolver owns
            // demand accounting; Runtime remains responsible for execution.
            [[nodiscard]] RequestResult apply(const Command& command);

            // Release every demand reference owned by this controller and reset
            // persistent application state. Runtime will use this during reset
            // and shutdown once it owns the controller lifecycle.
            void reset();
            [[nodiscard]] RequestState state() const;

        private:
            void acquire_once(core::ProductId product, bool& owned);
            void release_if_owned(core::ProductId product, bool& owned);

            core::DependencyResolver& resolver_;
            mutable std::mutex state_mutex_;
            RequestState state_;
            
            // These flags represent demand references owned specifically by this
            // controller. They prevent repeated commands from leaking resolver
            // reference counts and allow cancellation to release only Application
            // demand rather than another subsystem's demand.
            bool marker_depth_demand_owned_ = false;
            bool detection_demand_owned_ = false;
            bool segmentation_demand_owned_ = false;
            bool tracking_demand_owned_ = false;
            
            std::uint64_t next_tracking_query_revision_ = 1;
            std::uint64_t next_detection_query_revision_ = 1;
    };
}