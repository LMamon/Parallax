#include <parallax/tracking/track.hpp>
#include <parallax/tracking/tracker_ordering.hpp>

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

}