#include <parallax/perception/image_space_mapper.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

    parallax::perception::ImageSpaceMapper make_identity_mapper() {
        constexpr std::uint32_t width = 4;
        constexpr std::uint32_t height = 4;

        std::vector<float> map_x;
        std::vector<float> map_y;

        map_x.reserve(width * height);
        map_y.reserve(width * height);

        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                map_x.push_back(static_cast<float>(x));
                map_y.push_back(static_cast<float>(y));
            }
        }

        parallax::perception::ImageSpaceMapper mapper;
        EXPECT_TRUE(mapper.initialize(width, height, map_x, map_y));

        return mapper;
    }

}

TEST(ImageSpaceMapperTest, IdentityCalibrationMapsKnownPoint) {
    auto mapper = make_identity_mapper();
    cv::Point2f mapped{};

    ASSERT_TRUE(mapper.mapPoint({2.0F, 1.0F},
                                parallax::perception::ImageSpace::RgbLeft,
                                parallax::perception::ImageSpace::RectifiedLeft,
                                mapped));

    EXPECT_FLOAT_EQ(mapped.x, 2.0F);
    EXPECT_FLOAT_EQ(mapped.y, 1.0F);
}

TEST(ImageSpaceMapperTest, InvertsRectifiedToRgbCalibrationMap) {
    constexpr std::uint32_t width = 4;
    constexpr std::uint32_t height = 1;

    // Rectified pixels 0,1,2,3 sample source pixels 1,2,3,0.
    // Forward semantic mapping must therefore invert that relationship.    
    const std::vector<float> map_x{1.0F, 2.0F, 3.0F, 0.0F};
    const std::vector<float> map_y{0.0F, 0.0F, 0.0F, 0.0F};

    parallax::perception::ImageSpaceMapper mapper;
    ASSERT_TRUE(mapper.initialize(width, height, map_x, map_y));

    cv::Point2f mapped{};
    ASSERT_TRUE(mapper.mapPoint({1.0F, 0.0F},
                                parallax::perception::ImageSpace::RgbLeft,
                                parallax::perception::ImageSpace::RectifiedLeft,
                                mapped));

    EXPECT_FLOAT_EQ(mapped.x, 0.0F);
    EXPECT_FLOAT_EQ(mapped.y, 0.0F);

    ASSERT_TRUE(mapper.mapPoint({0.0F, 0.0F},
                                parallax::perception::ImageSpace::RgbLeft,
                                parallax::perception::ImageSpace::RectifiedLeft,
                                mapped));

    EXPECT_FLOAT_EQ(mapped.x, 3.0F);
    EXPECT_FLOAT_EQ(mapped.y, 0.0F);
}

TEST(ImageSpaceMapperTest, MapsBoundingRectangle) {
    auto mapper = make_identity_mapper();
    cv::Rect2f mapped{};

    ASSERT_TRUE(mapper.mapRect({1.0F, 1.0F, 2.0F, 2.0F},
                                parallax::perception::ImageSpace::RgbLeft,
                                parallax::perception::ImageSpace::RectifiedLeft,
                                mapped));

    EXPECT_FLOAT_EQ(mapped.x, 1.0F);
    EXPECT_FLOAT_EQ(mapped.y, 1.0F);
    EXPECT_FLOAT_EQ(mapped.width, 2.0F);
    EXPECT_FLOAT_EQ(mapped.height, 2.0F);
}

TEST(ImageSpaceMapperTest, SameSpaceMappingIsExplicitlySupported) {
    auto mapper = make_identity_mapper();
    cv::Point2f mapped{};

    ASSERT_TRUE(mapper.mapPoint({1.5F, 2.0F},
                                parallax::perception::ImageSpace::RgbLeft,
                                parallax::perception::ImageSpace::RgbLeft,
                                mapped));

    EXPECT_FLOAT_EQ(mapped.x, 1.5F);
    EXPECT_FLOAT_EQ(mapped.y, 2.0F);
}

TEST(ImageSpaceMapperTest, RejectsUnknownImageSpace) {
    auto mapper = make_identity_mapper();
    cv::Point2f mapped{};

    EXPECT_FALSE(mapper.mapPoint({1.0F, 1.0F},
                                 parallax::perception::ImageSpace::Unknown,
                                 parallax::perception::ImageSpace::RectifiedLeft,
                                 mapped));
}

TEST(ImageSpaceMapperTest, RejectsUnsupportedReverseMapping) {
    auto mapper = make_identity_mapper();
    cv::Point2f mapped{};

    EXPECT_FALSE(mapper.mapPoint({1.0F, 1.0F},
                                 parallax::perception::ImageSpace::RectifiedLeft,
                                 parallax::perception::ImageSpace::RgbLeft,
                                 mapped));
}

TEST(ImageSpaceMapperTest, RejectsUnmappedSourcePixel) {
    constexpr std::uint32_t width = 3;
    constexpr std::uint32_t height = 1;

    // No rectified sample points at source pixel x=2.
    const std::vector<float> map_x{0.0F, 0.0F, 1.0F};
    const std::vector<float> map_y{0.0F, 0.0F, 0.0F};

    parallax::perception::ImageSpaceMapper mapper;
    ASSERT_TRUE(mapper.initialize(width, height, map_x, map_y));
    cv::Point2f mapped{};

    EXPECT_FALSE(mapper.mapPoint({2.0F, 0.0F},
                                parallax::perception::ImageSpace::RgbLeft,
                                parallax::perception::ImageSpace::RectifiedLeft,
                                mapped));
}

TEST(ImageSpaceMapperTest, RejectsInvalidMapDimensions) {
    parallax::perception::ImageSpaceMapper mapper;

    const std::vector<float> map_x{0.0F};
    const std::vector<float> map_y{0.0F};

    EXPECT_FALSE(mapper.initialize(2, 2, map_x, map_y));
}