#include <parallax/application/request_controller.hpp>
#include <parallax/core/dependency_resolver.hpp>
#include <parallax/core/graph.hpp>

#include <gtest/gtest.h>

#include <string>

namespace {

    using parallax::application::Command;
    using parallax::application::CommandVerb;
    using parallax::application::CommandBehavior;
    using parallax::application::RequestController;
    using parallax::application::RequestStatus;
    using parallax::core::DemandSource;
    using parallax::core::DependencyResolver;
    using parallax::core::Graph;
    using parallax::core::ProductId;

    class RequestControllerTest : public ::testing::Test {
        protected:
            Graph graph_;
            DependencyResolver resolver_{graph_};
            RequestController controller_{resolver_};
    };

    TEST_F(RequestControllerTest, MarkerDepthAcquiresApplicationDemandOnlyOnce) {
        const Command command{CommandVerb::MarkerDepth, CommandBehavior::OneShot, ""};

        const auto first = controller_.apply(command);
        const auto second = controller_.apply(command);

        EXPECT_EQ(first.status, RequestStatus::Applied);
        EXPECT_EQ(second.status, RequestStatus::Applied);

        EXPECT_TRUE(controller_.state().marker_depth_requested);
        EXPECT_EQ(resolver_.demand(ProductId::MarkerDepth, DemandSource::Application), 1U);
    }

    TEST_F(RequestControllerTest, MarkerDepthDoesNotDisturbOtherDemandOwners) {
        resolver_.acquire(ProductId::MarkerDepth, DemandSource::RuntimeBaseline);

        resolver_.acquire(ProductId::MarkerDepth, DemandSource::FoxgloveSubscriber);

        const Command command{CommandVerb::MarkerDepth, CommandBehavior::OneShot, ""};

        (void)controller_.apply(command);

        EXPECT_EQ(resolver_.demand(ProductId::MarkerDepth, DemandSource::RuntimeBaseline), 1U);
        EXPECT_EQ(resolver_.demand(ProductId::MarkerDepth,DemandSource::FoxgloveSubscriber), 1U);
        EXPECT_EQ(resolver_.demand(ProductId::MarkerDepth, DemandSource::Application), 1U);

        controller_.reset();

        EXPECT_EQ(resolver_.demand(ProductId::MarkerDepth, DemandSource::Application), 0U);
        EXPECT_EQ(resolver_.demand(ProductId::MarkerDepth, DemandSource::RuntimeBaseline), 1U);
        EXPECT_EQ(resolver_.demand(ProductId::MarkerDepth, DemandSource::FoxgloveSubscriber), 1U);
    }

    TEST_F(RequestControllerTest, EmptyDetectionTargetDoesNotMutateStateOrDemand) {
        const Command command{CommandVerb::Detect, CommandBehavior::Persistent, ""};

        const auto result = controller_.apply(command);

        EXPECT_EQ(result.status, RequestStatus::Invalid);
        EXPECT_FALSE(controller_.state().detection_requested);

        EXPECT_TRUE(controller_.state().detection_target.empty());
        EXPECT_EQ(resolver_.demand(ProductId::Detection, DemandSource::Application), 0U);
        EXPECT_EQ(controller_.state().detection_query_revision, 0U);
    }

    TEST_F(RequestControllerTest, DetectionIntentPersistsAndAcquiresDemand) {
        const Command command{CommandVerb::Detect, CommandBehavior::Persistent, "person"};

        const auto result = controller_.apply(command);

        EXPECT_EQ(result.status, RequestStatus::Applied);
        EXPECT_TRUE(controller_.state().detection_requested);
        EXPECT_EQ(controller_.state().detection_target, "person");
        EXPECT_EQ(resolver_.demand(ProductId::Detection, DemandSource::Application), 1U);
        EXPECT_EQ(controller_.state().detection_query_revision, 1U);
    }

    TEST_F(RequestControllerTest, ReissuingDetectionReplacesTargetAndAdvancesRevisionWithoutLeakingDemand) {
        (void)controller_.apply(Command{CommandVerb::Detect, CommandBehavior::Persistent, "person"});
        const auto first_revision = controller_.state().detection_query_revision;

        (void)controller_.apply(Command{CommandVerb::Detect, CommandBehavior::Persistent, "vehicle"});
        const auto second_revision = controller_.state().detection_query_revision;

        EXPECT_TRUE(controller_.state().detection_requested);
        EXPECT_EQ(controller_.state().detection_target, "vehicle");
        EXPECT_GT(second_revision, first_revision);
        EXPECT_EQ(resolver_.demand(ProductId::Detection, DemandSource::Application), 1U);
    }

    TEST_F(RequestControllerTest, EmptyTrackingTargetDoesNotMutateStateOrDemand) {
        const auto result = controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, ""});

        EXPECT_EQ(result.status, RequestStatus::Invalid);

        EXPECT_FALSE(controller_.state().tracking_requested);
        EXPECT_TRUE(controller_.state().tracking_target.empty());

        EXPECT_EQ(resolver_.demand(ProductId::Track2D, DemandSource::Application), 0U);
    }

    TEST_F(RequestControllerTest, TrackingTargetReplacementDoesNotLeakDemand) {
        const auto first = controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "person"});
        const auto first_revision = controller_.state().tracking_query_revision;
        const auto second = controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "vehicle"});

        const auto state = controller_.state();

        EXPECT_EQ(first.status, RequestStatus::Applied);
        EXPECT_EQ(second.status, RequestStatus::Applied);

        EXPECT_TRUE(controller_.state().tracking_requested);

        EXPECT_EQ(controller_.state().tracking_target, "vehicle");
        EXPECT_GT(state.tracking_query_revision, first_revision);
        EXPECT_EQ(resolver_.demand(ProductId::Track2D, DemandSource::Application), 1U);
    }

    TEST_F(RequestControllerTest, StopTrackingReleasesOnlyApplicationTrackingDemand) {
        resolver_.acquire(ProductId::Track2D, DemandSource::FoxgloveSubscriber);

        (void)controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "person"});

        const auto result = controller_.apply(Command{CommandVerb::StopTracking, CommandBehavior::OneShot, ""});

        EXPECT_EQ(result.status, RequestStatus::Applied);

        EXPECT_FALSE(controller_.state().tracking_requested);
        EXPECT_TRUE(controller_.state().tracking_target.empty());

        EXPECT_EQ(resolver_.demand(ProductId::Track2D, DemandSource::Application), 0U);
        EXPECT_EQ(resolver_.demand(ProductId::Track2D, DemandSource::FoxgloveSubscriber), 1U);
        EXPECT_EQ(controller_.state().tracking_query_revision, 0U);
    }

    TEST_F(RequestControllerTest, RepeatedStopTrackingDoesNotUnderflowDemand) {
        (void)controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "person"});
        (void)controller_.apply(Command{CommandVerb::StopTracking, CommandBehavior::OneShot, ""});
        (void)controller_.apply(Command{CommandVerb::StopTracking, CommandBehavior::OneShot, ""});

        EXPECT_EQ(resolver_.demand(ProductId::Track2D, DemandSource::Application), 0U);
        EXPECT_EQ(resolver_.total_demand(ProductId::Track2D), 0U);
    }

    TEST_F(RequestControllerTest, ResetReleasesAllApplicationDemandAndClearsState) {
        (void)controller_.apply(Command{CommandVerb::MarkerDepth, CommandBehavior::OneShot, ""});
        (void)controller_.apply(Command{CommandVerb::Detect, CommandBehavior::Persistent, "person"});
        (void)controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "person"});
        (void)controller_.apply(Command{CommandVerb::Segment, CommandBehavior::OneShot, "cup"});

        controller_.reset();

        EXPECT_EQ(resolver_.demand(ProductId::MarkerDepth, DemandSource::Application), 0U);
        EXPECT_EQ(resolver_.demand(ProductId::Detection, DemandSource::Application), 0U);
        EXPECT_EQ(resolver_.demand(ProductId::Track2D, DemandSource::Application), 0U);
        EXPECT_EQ(resolver_.demand(ProductId::Segmentation, DemandSource::Application), 0U);
        
        const auto& state = controller_.state();
        
        EXPECT_FALSE(state.marker_depth_requested);
        EXPECT_FALSE(state.detection_requested);
        EXPECT_TRUE(state.detection_target.empty());
        EXPECT_FALSE(state.tracking_requested);
        EXPECT_TRUE(state.tracking_target.empty());
        EXPECT_FALSE(state.segmentation_requested);
        EXPECT_TRUE(state.segmentation_target.empty());
        EXPECT_EQ(state.tracking_query_revision, 0U);
    }

    TEST_F(RequestControllerTest, ResetDoesNotReuseDetectionQueryRevision) {
        (void)controller_.apply(Command{CommandVerb::Detect, CommandBehavior::Persistent, "person"});

        const auto first_revision = controller_.state().detection_query_revision;

        controller_.reset();
        EXPECT_EQ(controller_.state().detection_query_revision, 0U);

        (void)controller_.apply(Command{CommandVerb::Detect, CommandBehavior::Persistent, "vehicle"});
        EXPECT_GT(controller_.state().detection_query_revision, first_revision);
    }

    TEST_F(RequestControllerTest, DetectionDoesNotAcquireSegmentationDemand) {
        const auto result = controller_.apply(Command{CommandVerb::Detect, CommandBehavior::Persistent, "person"});

        EXPECT_EQ(result.status, RequestStatus::Applied);

        const auto state = controller_.state();

        EXPECT_TRUE(state.detection_requested);
        EXPECT_FALSE(state.segmentation_requested);

        EXPECT_EQ(resolver_.demand(ProductId::Detection, DemandSource::Application), 1U);
        EXPECT_EQ(resolver_.demand(ProductId::Segmentation, DemandSource::Application), 0U);
    }

    TEST_F(RequestControllerTest, SegmentationAcquiresDetectionAndSegmentationDemand) {
        const auto result = controller_.apply(Command{CommandVerb::Segment, CommandBehavior::OneShot, "person"});

        EXPECT_EQ(result.status, RequestStatus::Applied);

        const auto state = controller_.state();

        EXPECT_TRUE(state.detection_requested);
        EXPECT_TRUE(state.segmentation_requested);

        EXPECT_EQ(state.detection_target, "person");
        EXPECT_EQ(state.segmentation_target, "person");

        EXPECT_EQ(resolver_.demand(ProductId::Detection, DemandSource::Application), 1U);
        EXPECT_EQ(resolver_.demand(ProductId::Segmentation, DemandSource::Application), 1U);
    }

    TEST_F(RequestControllerTest, RepeatedSegmentationRequestDoesNotLeakDemand) {
        (void)controller_.apply(Command{CommandVerb::Segment, CommandBehavior::OneShot, "person"});

        const auto first_revision = controller_.state().detection_query_revision;
        (void)controller_.apply(Command{CommandVerb::Segment, CommandBehavior::OneShot, "cup"});

        const auto state = controller_.state();

        EXPECT_EQ(state.detection_target, "cup");
        EXPECT_EQ(state.segmentation_target, "cup");

        EXPECT_GT(state.detection_query_revision, first_revision);
        EXPECT_EQ(resolver_.demand(ProductId::Detection, DemandSource::Application), 1U);
        EXPECT_EQ(resolver_.demand(ProductId::Segmentation, DemandSource::Application), 1U);
    }

    TEST_F(RequestControllerTest, ResetReleasesSegmentationDemand) {
        (void)controller_.apply(Command{CommandVerb::Segment, CommandBehavior::OneShot, "person"});

        controller_.reset();

        EXPECT_EQ(resolver_.demand(ProductId::Detection, DemandSource::Application), 0U);
        EXPECT_EQ(resolver_.demand(ProductId::Segmentation, DemandSource::Application), 0U);

        const auto state = controller_.state();

        EXPECT_FALSE(state.detection_requested);
        EXPECT_FALSE(state.segmentation_requested);
        EXPECT_TRUE(state.detection_target.empty());
        EXPECT_TRUE(state.segmentation_target.empty());
    }

    TEST_F(RequestControllerTest, RepeatedTrackingTargetPreservesRevisionAndDemand) {
        const auto first = controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "person"});
        const auto first_state = controller_.state();

        const auto second = controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "person"});
        const auto second_state = controller_.state();

        EXPECT_EQ(first.status, RequestStatus::Applied);
        EXPECT_EQ(second.status, RequestStatus::Applied);

        EXPECT_EQ(second_state.tracking_query_revision, first_state.tracking_query_revision);
        EXPECT_EQ(resolver_.demand(ProductId::Track2D, DemandSource::Application), 1U);
    }

    TEST_F(RequestControllerTest, TrackingRevisionIsNotReusedAfterStop) {
        (void)controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "person"});
        const auto first_revision = controller_.state().tracking_query_revision;

        (void)controller_.apply(Command{CommandVerb::StopTracking, CommandBehavior::OneShot, ""});
        EXPECT_EQ(controller_.state().tracking_query_revision, 0U);
        
        (void)controller_.apply(Command{CommandVerb::Track, CommandBehavior::Persistent, "person"});
        EXPECT_GT(controller_.state().tracking_query_revision, first_revision);
    }
}