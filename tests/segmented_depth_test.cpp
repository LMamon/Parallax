#include <gtest/gtest.h>

#include <parallax/perception/segmented_depth.hpp>

using namespace parallax;

TEST(SegmentedDepth, RequiresMaskSupportedObjectFromSameObservation) {
    perception::SegmentedDepth segmented{};
    segmented.query = "person";
    segmented.query_revision = 7;
    segmented.source_observation = {core::SourceId::StereoCamera, 42};

    auto& object = segmented.object;
    object.label = "person";
    object.query_revision = 7;
    object.image_space = perception::ImageSpace::RgbLeft;
    object.semantic_observation = segmented.source_observation;
    object.metric_observation = segmented.source_observation;
    object.coordinate_frame = "camera_left_optical";
    object.geometry = perception::Object3DGeometry::Surface;
    object.method = perception::Object3DMethod::StereoMask;
    object.depth_m = 2.0F;
    object.position_m = {0.0F, 0.0F, 2.0F};

    EXPECT_TRUE(segmented.valid());

    object.method = perception::Object3DMethod::StereoRoi;
    EXPECT_FALSE(segmented.valid());
}
