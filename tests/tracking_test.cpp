#include <parallax/tracking/track.hpp>

#include <gtest/gtest.h>

namespace {

    using parallax::core::SourceId;
    using parallax::core::SourceObservation;
    using parallax::perception::ImageSpace;
    using parallax::tracking::Track2D;
    using parallax::tracking::TrackLifecycle;

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

}