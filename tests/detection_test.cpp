#include <parallax/core/product.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/image_space.hpp>

#include <gtest/gtest.h>

#include <memory>

namespace {

    using parallax::core::ProductId;
    using parallax::core::ProductMetadata;
    using parallax::core::SourceId;
    using parallax::core::SourceObservation;
    using parallax::perception::DetectionSet;
    using parallax::perception::ImageSpace;

    TEST(DetectionSetTest, EmptyResultCanStillBeValid) {
        const DetectionSet result{"person",
                                  1,
                                  ImageSpace::RgbLeft,
                                  {}, {}, {}};

        EXPECT_TRUE(result.valid());
        EXPECT_TRUE(result.empty());
    }

    TEST(DetectionSetTest, UnknownImageSpaceIsInvalid) {
        const DetectionSet result{"person", 
                                  1,
                                  ImageSpace::Unknown,
                                  {}, {}, {}};

        EXPECT_FALSE(result.valid());
    }

    TEST(DetectionSetTest, ParallelOutputsMustHaveMatchingSizes) {
        const DetectionSet result{"person",
                                  1,
                                  ImageSpace::RgbLeft,
                                  {cv::Rect2f{10.0F, 20.0F, 30.0F, 40.0F}},
                                  {}, {}};

        EXPECT_FALSE(result.valid());
    }

    TEST(DetectionSetTest, QueryRevisionDistinguishesPromptReplacement) {
        const DetectionSet person{"person",
                                  4,
                                  ImageSpace::RgbLeft,
                                  {}, {}, {}};

        const DetectionSet cup{"cup",
                                5,
                                ImageSpace::RgbLeft,
                                {}, {}, {}};

        EXPECT_NE(person.query_revision, cup.query_revision);
        EXPECT_NE(person.query, cup.query);
    }

    TEST(DetectionSetTest, ProductMetadataOwnsCameraProvenance) {
        ProductMetadata metadata{};
        metadata.observation = SourceObservation{SourceId::StereoCamera, 120};
        metadata.valid = true;

        std::shared_ptr<const DetectionSet> result = std::make_shared<DetectionSet>(
                                                            DetectionSet{"person",
                                                            7,
                                                            ImageSpace::RgbLeft,
                                                            {cv::Rect2f{10.0F, 20.0F, 30.0F, 40.0F}},
                                                            {0.92F},
                                                            {0}});

        auto product = parallax::core::make_product(ProductId::Detection,
                                                    metadata,
                                                    std::move(result));

        ASSERT_TRUE(product.valid());
        EXPECT_EQ(product.metadata.observation.source, SourceId::StereoCamera);
        EXPECT_EQ(product.metadata.observation.sequence, 120U);
        ASSERT_EQ(product.payload->size(), 1U);
        EXPECT_EQ(product.payload->query_revision, 7U);
        EXPECT_EQ(product.payload->image_space, ImageSpace::RgbLeft);
    }

    TEST(DetectionSetTest, SameObservationDoesNotImplySameImageSpace) {
        ProductMetadata metadata{};
        metadata.observation = SourceObservation{SourceId::StereoCamera, 42};
        metadata.valid = true;

        const std::shared_ptr<const DetectionSet> rgb_detection = std::make_shared<DetectionSet>(
                                                                  DetectionSet{"person",
                                                                  1,
                                                                  ImageSpace::RgbLeft,
                                                                  {}, {}, {}});

        const std::shared_ptr<const DetectionSet> rectified_detection = std::make_shared<DetectionSet>(
                                                                        DetectionSet{"person",
                                                                        1,
                                                                        ImageSpace::RectifiedLeft,
                                                                        {}, {}, {}});

        const auto rgb_product = parallax::core::make_product(ProductId::Detection,
                                                              metadata,
                                                              rgb_detection);

        const auto rectified_product = parallax::core::make_product(ProductId::Detection,
                                                                    metadata,
                                                                    rectified_detection);

        EXPECT_EQ(rgb_product.metadata.observation.source, rectified_product.metadata.observation.source);
        EXPECT_EQ(rgb_product.metadata.observation.sequence, rectified_product.metadata.observation.sequence);
        EXPECT_NE(rgb_product.payload->image_space, rectified_product.payload->image_space);
    }
}