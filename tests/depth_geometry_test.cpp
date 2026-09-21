#include <gtest/gtest.h>

#include <parallax/stereo/depth_geometry.hpp>

#include <array>
#include <limits>

namespace {

    TEST(DepthGeometryTest, BackProjectsPreviewGeometry) {
        constexpr std::uint32_t width = 4;
        constexpr std::uint32_t height = 3;
        const float depth[width * height] = {
            2.0F, 2.0F, 2.0F, 2.0F,
            2.0F, 2.0F, 2.0F, 2.0F,
            2.0F, 2.0F, 2.0F, 2.0F
        };
        const std::array<double, 12> p{
            2.0, 0.0, 1.0, 0.0,
            0.0, 2.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        };

        const auto points = parallax::stereo::backProjectDepthSamples(depth, width, height, 2, p);
        ASSERT_EQ(points.size(), 4U);
        EXPECT_FLOAT_EQ(points[0][0], -1.0F);
        EXPECT_FLOAT_EQ(points[0][1], -1.0F);
        EXPECT_FLOAT_EQ(points[0][2], 2.0F);
        EXPECT_FLOAT_EQ(points[1][0], 1.0F);
        EXPECT_FLOAT_EQ(points[1][1], -1.0F);
        EXPECT_FLOAT_EQ(points[2][0], -1.0F);
        EXPECT_FLOAT_EQ(points[2][1], 1.0F);
    }

    TEST(DepthGeometryTest, SkipsInvalidDepthOnTheSamplingGrid) {
        const float depth[3] = {
            std::numeric_limits<float>::quiet_NaN(), 3.0F, 4.0F
        };
        const std::array<double, 12> p{
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        };

        const auto points = parallax::stereo::backProjectDepthSamples(depth, 3, 1, 2, p);
        ASSERT_EQ(points.size(), 1U);
        EXPECT_FLOAT_EQ(points.front()[0], 8.0F);
        EXPECT_FLOAT_EQ(points.front()[2], 4.0F);
    }

    TEST(DepthGeometryTest, RejectsInvalidProjection) {
        const float depth[1] = {2.0F};
        std::array<double, 12> p{};
        p[5] = 1.0;
        p[10] = 1.0;
        EXPECT_TRUE(parallax::stereo::backProjectDepthSamples(depth, 1, 1, 1, p).empty());
    }

}
