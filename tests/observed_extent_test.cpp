#include <gtest/gtest.h>
#include <parallax/perception/observed_extent.hpp>

#include <array>
#include <vector>

namespace {

    TEST(ObservedExtentTest, RejectsSparseSupport) {
        std::vector<std::array<float, 3>> points(7, {0.0F, 0.0F, 2.0F});
        parallax::perception::ObservedExtent3D extent{};
        EXPECT_FALSE(parallax::perception::estimateObservedExtent(points, extent));
    }

    TEST(ObservedExtentTest, TrimsIsolatedStereoOutliers) {
        std::vector<std::array<float, 3>> points;
        for (int z = 0; z < 5; ++z)
            for (int y = 0; y < 5; ++y)
                for (int x = 0; x < 5; ++x)
                    points.push_back({-0.20F + 0.10F * x,
                                      -0.10F + 0.05F * y,
                                       1.95F + 0.025F * z});

        points.push_back({12.0F, -9.0F, 30.0F});

        parallax::perception::ObservedExtent3D extent{};
        ASSERT_TRUE(parallax::perception::estimateObservedExtent(points, extent));
        ASSERT_TRUE(extent.valid());

        EXPECT_LT(extent.size_m[0], 1.0F);
        EXPECT_LT(extent.size_m[1], 1.0F);
        EXPECT_LT(extent.size_m[2], 1.0F);
        EXPECT_NEAR(extent.center_m[0], 0.0F, 0.06F);
        EXPECT_NEAR(extent.center_m[1], 0.0F, 0.04F);
        EXPECT_NEAR(extent.center_m[2], 2.0F, 0.05F);
    }

    TEST(ObservedExtentTest, DoesNotInventMissingThickness) {
        std::vector<std::array<float, 3>> points;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                points.push_back({0.02F * x, 0.02F * y, 2.0F});

        parallax::perception::ObservedExtent3D extent{};
        EXPECT_FALSE(parallax::perception::estimateObservedExtent(points, extent));
    }

}
