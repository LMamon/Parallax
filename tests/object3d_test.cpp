#include <parallax/perception/object3d.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <limits>

namespace {
    parallax::perception::Object3D make_valid_object() {
        using namespace parallax;

        perception::Object3D object;
        object.label = "cup";
        object.query_revision = 1;
        object.semantic_confidence = 0.91F;

        object.image_box = cv::Rect2f{100.0F, 80.0F, 40.0F, 60.0F};
        object.image_space = perception::ImageSpace::RgbLeft;

        object.position_m = {0.1F, -0.2F, 1.5F};
        object.depth_m = 1.5F;
        object.coordinate_frame = "camera_left_optical";

        object.geometry = perception::Object3DGeometry::Point;
        object.method = perception::Object3DMethod::StereoRoi;

        object.semantic_observation = {core::SourceId::StereoCamera, 42};
        object.metric_observation = {core::SourceId::StereoCamera, 42};

        object.association_timestamp = std::chrono::steady_clock::now();
        object.source_time_delta = std::chrono::steady_clock::duration::zero();

        object.support_quality = 0.8F;

        return object;
    }
}

TEST(Object3DTest, DetectionDoesNotRequirePersistentTrackIdentity) {
    const auto object = make_valid_object();

    EXPECT_TRUE(object.valid());
    EXPECT_FALSE(object.persistent());
    EXPECT_EQ(object.track_id, 0U);
}

TEST(Object3DTest, PersistentTargetIdentityIsOptional) {
    auto object = make_valid_object();
    object.track_id = 7;

    EXPECT_TRUE(object.valid());
    EXPECT_TRUE(object.persistent());
    EXPECT_EQ(object.track_id, 7U);
}

TEST(Object3DTest, PreservesSemanticAndMetricProvenanceIndependently) {
    auto object = make_valid_object();

    object.semantic_observation = {parallax::core::SourceId::StereoCamera, 100};
    object.metric_observation = {parallax::core::SourceId::StereoCamera, 99};

    EXPECT_TRUE(object.valid());
    EXPECT_EQ(object.semantic_observation.sequence, 100U);
    EXPECT_EQ(object.metric_observation.sequence, 99U);
}

TEST(Object3DTest, RejectsInvalidMetricGeometry) {
    auto object = make_valid_object();
    object.depth_m = std::numeric_limits<float>::quiet_NaN();

    EXPECT_FALSE(object.valid());
}

TEST(Object3DTest, RejectsMissingCoordinateFrame) {
    auto object = make_valid_object();
    object.coordinate_frame.clear();

    EXPECT_FALSE(object.valid());
}

TEST(Object3DSetTest, EmptyDetectionSetCanStillBeValid) {
    parallax::perception::Object3DSet objects;
    objects.query = "cup";
    objects.query_revision = 3;

    EXPECT_TRUE(objects.valid());
    EXPECT_TRUE(objects.empty());
}

TEST(Object3DSetTest, RejectsInvalidContainedObject) {
    parallax::perception::Object3DSet objects;
    objects.query = "cup";
    objects.query_revision = 3;

    auto object = make_valid_object();
    object.method = parallax::perception::Object3DMethod::Unknown;
    objects.objects.push_back(object);

    EXPECT_FALSE(objects.valid());
}