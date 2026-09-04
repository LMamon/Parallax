#include <parallax/tracking/track.hpp>
#include <parallax/tracking/tracker_ordering.hpp>
#include <parallax/core/dependency_resolver.hpp>
#include <parallax/core/graph.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/tracking/single_target_producer.hpp>
#include <parallax/core/execution_context.hpp>
#include <parallax/perception/detection.hpp>

#include <chrono>
#include <memory>
#include <gtest/gtest.h>

namespace {

    using parallax::core::SourceId;
    using parallax::core::SourceObservation;
    using parallax::perception::ImageSpace;
    using parallax::tracking::Track2D;
    using parallax::tracking::TrackLifecycle;
    using parallax::tracking::TrackerFrameDecision;
    using parallax::tracking::evaluate_tracker_frame;

    TEST(Track2DTest, DefaultTrackIsIdleAndInvalid) {
        const Track2D track{};

        EXPECT_EQ(track.lifecycle, TrackLifecycle::Idle);
        EXPECT_FALSE(track.valid());
    }

    TEST(Track2DTest, TrackingStateCanBeValid) {
        Track2D track{};

        track.track_id = 1;
        track.target_query = "person";
        track.target_revision = 4;
        track.box = cv::Rect2f{100.0F, 50.0F, 200.0F, 300.0F};
        track.quality = 0.92F;
        track.lifecycle = TrackLifecycle::Tracking;
        track.image_space = ImageSpace::RgbLeft;

        EXPECT_TRUE(track.valid());
    }

    TEST(Track2DTest, PreservesDetectorAndTrackerProvenanceSeparately) {
        Track2D track{};

        track.source_observation = SourceObservation{SourceId::StereoCamera, 120};
        track.last_detector_observation = SourceObservation{SourceId::StereoCamera, 100};
        track.last_tracker_observation = SourceObservation{SourceId::StereoCamera, 120};

        EXPECT_EQ(track.source_observation.sequence, 120);
        EXPECT_EQ(track.last_detector_observation.sequence, 100);
        EXPECT_EQ(track.last_tracker_observation.sequence, 120);
    }

    TEST(Track2DTest, LifecycleDoesNotDependOnProductMetadataValidity) {
        Track2D track{};

        track.track_id = 7;
        track.target_query = "person";
        track.target_revision = 2;
        track.lifecycle = TrackLifecycle::Lost;
        track.image_space = ImageSpace::RgbLeft;

        EXPECT_TRUE(track.valid());
        EXPECT_EQ(track.lifecycle, TrackLifecycle::Lost);
    }

    TEST(TrackerOrderingTest, FirstObservationEstablishesClock) {
        const SourceObservation none{};
        const SourceObservation candidate{SourceId::StereoCamera, 10};

        const auto result = evaluate_tracker_frame(none, candidate);

        EXPECT_TRUE(result.accepted());
        EXPECT_FALSE(result.requires_reset());
        EXPECT_EQ(result.skipped, 0U);
    }

    TEST(TrackerOrderingTest, SequentialObservationIsAccepted) {
        const SourceObservation previous{SourceId::StereoCamera, 10};
        const SourceObservation candidate{SourceId::StereoCamera, 11};
        const auto result = evaluate_tracker_frame(previous, candidate);

        EXPECT_TRUE(result.accepted());
        EXPECT_EQ(result.skipped, 0U);
    }

    TEST(TrackerOrderingTest, DuplicateObservationIsRejected) {
        const SourceObservation observation{SourceId::StereoCamera, 10};
        const auto result = evaluate_tracker_frame(observation, observation);

        EXPECT_EQ(result.decision, TrackerFrameDecision::RejectDuplicate);
        EXPECT_FALSE(result.requires_reset());
    }

    TEST(TrackerOrderingTest, OlderObservationIsRejected) {
        const SourceObservation previous{SourceId::StereoCamera, 10};
        const SourceObservation candidate{SourceId::StereoCamera, 9};
        const auto result = evaluate_tracker_frame(previous, candidate);

        EXPECT_EQ(result.decision, TrackerFrameDecision::RejectOlder);
        EXPECT_FALSE(result.requires_reset());
    }

    TEST(TrackerOrderingTest, SmallGapSkipsIntermediateObservations) {
        const SourceObservation previous{SourceId::StereoCamera, 10};
        const SourceObservation candidate{SourceId::StereoCamera, 13};
        const auto result = evaluate_tracker_frame(previous, candidate);

        EXPECT_TRUE(result.accepted());
        EXPECT_EQ(result.skipped, 2U);
    }

    TEST(TrackerOrderingTest, OversizedGapRequiresReset) {
        const SourceObservation previous{SourceId::StereoCamera, 10};
        const SourceObservation candidate{SourceId::StereoCamera, 14};
        const auto result = evaluate_tracker_frame(previous, candidate);

        EXPECT_EQ(result.decision, TrackerFrameDecision::ResetGap);
        EXPECT_TRUE(result.requires_reset());
        EXPECT_EQ(result.skipped, 3U);
    }

    TEST(TrackerOrderingTest, SourceChangeRequiresReset) {
        const SourceObservation previous{SourceId::StereoCamera, 10};
        const SourceObservation candidate{SourceId::Rplidar, 11};
        const auto result = evaluate_tracker_frame(previous, candidate);

        EXPECT_EQ(result.decision, TrackerFrameDecision::RejectSource);
        EXPECT_TRUE(result.requires_reset());
    }

    TEST(SingleTargetProducerTest, RgbIsTheContinuousGraphInput) {
        parallax::core::Graph graph;
        parallax::core::DependencyResolver resolver{graph};
        parallax::core::ProductStore products;

        parallax::tracking::SingleTargetProducer producer{products, resolver};

        ASSERT_EQ(producer.inputs().size(), 1U);
        EXPECT_EQ(producer.inputs().front(), parallax::core::ProductId::RgbLeft);
        ASSERT_EQ(producer.compatible_inputs().size(), 1U);
        EXPECT_EQ(producer.compatible_inputs().front().product, parallax::core::ProductId::RgbLeft);
        EXPECT_EQ(producer.compatible_inputs().front().history_capacity, 2U);
    }

    TEST(SingleTargetProducerTest, TargetAcquiresOneDetectorReference) {
        parallax::core::Graph graph;
        parallax::core::DependencyResolver resolver{graph};
        parallax::core::ProductStore products;

        parallax::tracking::SingleTargetProducer producer{products, resolver};

        ASSERT_TRUE(producer.setTarget("person", 1));
        EXPECT_TRUE(producer.needsDetection());
        EXPECT_EQ(resolver.demand(parallax::core::ProductId::Detection, parallax::core::DemandSource::InternalDependent), 1U);
    }

    TEST(SingleTargetProducerTest, RepeatedTargetDoesNotLeakDetectorDemand) {
        parallax::core::Graph graph;
        parallax::core::DependencyResolver resolver{graph};
        parallax::core::ProductStore products;

        parallax::tracking::SingleTargetProducer producer{products, resolver};

        ASSERT_TRUE(producer.setTarget("person", 1));
        ASSERT_TRUE(producer.setTarget("person", 1));
        EXPECT_EQ(resolver.demand(parallax::core::ProductId::Detection, parallax::core::DemandSource::InternalDependent), 1U);
    }

    TEST(SingleTargetProducerTest, TargetReplacementKeepsDetectorDemandBounded) {
        parallax::core::Graph graph;
        parallax::core::DependencyResolver resolver{graph};
        parallax::core::ProductStore products;

        parallax::tracking::SingleTargetProducer producer{products, resolver};

        ASSERT_TRUE(producer.setTarget("person", 1));
        ASSERT_TRUE(producer.setTarget("vehicle", 2));

        EXPECT_EQ(producer.targetQuery(), "vehicle");
        EXPECT_EQ(producer.targetRevision(), 2U);
        EXPECT_EQ(resolver.demand(parallax::core::ProductId::Detection, parallax::core::DemandSource::InternalDependent), 1U);
    }

    TEST(SingleTargetProducerTest, ResetReleasesTrackerOwnedDetectorDemand) {
        parallax::core::Graph graph;
        parallax::core::DependencyResolver resolver{graph};
        parallax::core::ProductStore products;

        parallax::tracking::SingleTargetProducer producer{products, resolver};

        ASSERT_TRUE(producer.setTarget("person", 1));
        ASSERT_EQ(resolver.demand(parallax::core::ProductId::Detection, parallax::core::DemandSource::InternalDependent), 1U);

        producer.reset();

        EXPECT_FALSE(producer.needsDetection());
        EXPECT_TRUE(producer.targetQuery().empty());
        EXPECT_EQ(resolver.demand(parallax::core::ProductId::Detection, parallax::core::DemandSource::InternalDependent), 0U);
    }

    TEST(SingleTargetProducerTest, ExposesTrackingState) {
        parallax::core::Graph graph;
        parallax::core::DependencyResolver resolver{graph};
        parallax::core::ProductStore products;

        parallax::tracking::SingleTargetProducer producer{products, resolver};

        EXPECT_FALSE(producer.tracking());
        EXPECT_FALSE(producer.track().valid());

        ASSERT_TRUE(producer.setTarget("person", 1));

        EXPECT_FALSE(producer.tracking());
        EXPECT_TRUE(producer.needsDetection());

        const auto& track = producer.track();

        EXPECT_EQ(track.target_query, "person");
        EXPECT_EQ(track.target_revision, 1U);
        EXPECT_EQ(track.lifecycle, parallax::tracking::TrackLifecycle::Reacquiring);
    }

    TEST(SingleTargetProducerTest, RejectsDetectionFromOldTargetRevision) {
        parallax::core::Graph graph;
        parallax::core::DependencyResolver resolver{graph};
        parallax::core::ProductStore products;
        parallax::core::ExecutionContext context;

        parallax::tracking::SingleTargetProducer producer{products, resolver};

        ASSERT_TRUE(producer.setTarget("person", 2));

        auto detections = std::make_shared<parallax::perception::DetectionSet>();

        detections->query = "person";
        detections->query_revision = 1;
        detections->image_space = parallax::perception::ImageSpace::RgbLeft;

        detections->boxes.push_back(cv::Rect2f{100.0F,
                                               100.0F,
                                               80.0F,
                                               120.0F});

        detections->scores.push_back(0.9F);
        detections->labels.push_back(0);

        parallax::core::ProductMetadata metadata{};
        metadata.observation = {parallax::core::SourceId::StereoCamera, 10};

        metadata.timestamp = std::chrono::steady_clock::now();

        metadata.production_timestamp = metadata.timestamp;
        metadata.valid = true;
        products.publish(parallax::core::make_product<parallax::perception::DetectionSet>(
                                                      parallax::core::ProductId::Detection,
                                                      metadata, std::move(detections),
                                                      parallax::core::CompletionHandle::cpu_ready()));

        EXPECT_EQ(producer.submit(context), parallax::core::SubmitResult::NoWork);
        EXPECT_TRUE(producer.needsDetection());
        EXPECT_FALSE(producer.tracking());
    }

    TEST(SingleTargetProducerTest, DetectorDemandIsVisibleInMetrics) {
        parallax::core::Graph graph;
        parallax::core::DependencyResolver resolver{graph};
        parallax::core::ProductStore products;

        parallax::tracking::SingleTargetProducer producer{products, resolver};

        ASSERT_TRUE(producer.setTarget("person", 1));

        const auto metrics = producer.metrics();

        EXPECT_EQ(metrics.detector_refreshes, 1U);
        EXPECT_EQ(metrics.reacquisition_requests, 0U);
    }
}