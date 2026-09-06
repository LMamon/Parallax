#include <parallax/visualization/publisher.hpp>

#include <gtest/gtest.h>

TEST(Object3DVisualizationTest, UsesInchesBelowOneFoot) {
    EXPECT_EQ(parallax::visualization::formatDepthForDisplay(0.20F), "7.9 in");
}

TEST(Object3DVisualizationTest, UsesFeetFromOneToUnderThreeFeet) {
    EXPECT_EQ(parallax::visualization::formatDepthForDisplay(0.6096F), "2.0 ft");
}

TEST(Object3DVisualizationTest, UsesMetersAtThreeFeetAndBeyond) {
    EXPECT_EQ(parallax::visualization::formatDepthForDisplay(1.0F),"1.00 m");
}

TEST(Object3DVisualizationTest, RejectsInvalidDepth) {
    EXPECT_TRUE(parallax::visualization::formatDepthForDisplay(-1.0F).empty());
}