#include <parallax/core/product.hpp>
#include <parallax/perception/detection.hpp>

#include <gtest/gtest.h>

#include <memory>

namespace {

    using parallax::core::ProductId;
    using parallax::core::ProductMetadata;
    using parallax::core::SourceId;
    using parallax::core::SourceObservation;
    using parallax::perception::DetectionSet;

    TEST(DetectionSetTest, EmptyResultCanStillBeValid) {
        const DetectionSet result{"person", 1, {}, {}, {}};

        EXPECT_TRUE(result.valid());
        EXPECT_TRUE(result.empty());
    }

    TEST(DetectionSetTest, ParallelOutputsMustHaveMatchingSizes) {
        DetectionSet result{"person", 1,
                            {cv::Rect2f{10.0F, 20.0F, 30.0F, 40.0F}},
                            {},
                            {}};

        EXPECT_FALSE(result.valid());
    }

    TEST(DetectionSetTest, QueryRevisionDistinguishesPromptReplacement) {
        const DetectionSet person{"person", 4, {}, {}, {}};
        const DetectionSet cup{"cup", 5, {}, {}, {}};

        EXPECT_NE(person.query_revision, cup.query_revision);
        EXPECT_NE(person.query, cup.query);
    }

    TEST(DetectionSetTest, ProductMetadataOwnsCameraProvenance) {
        ProductMetadata metadata{};
        metadata.observation = SourceObservation{SourceId::StereoCamera, 120};
        metadata.valid = true;

        std::shared_ptr<const DetectionSet> result = std::make_shared<DetectionSet>(
            DetectionSet{"person", 7, 
                        {cv::Rect2f{10.0F, 20.0F, 30.0F, 40.0F}},
                        {0.92F},
                        {0}});

        auto product = parallax::core::make_product(ProductId::Detection, metadata, std::move(result));

        ASSERT_TRUE(product.valid());
        EXPECT_EQ(product.metadata.observation.source, SourceId::StereoCamera);
        EXPECT_EQ(product.metadata.observation.sequence, 120U);
        ASSERT_EQ(product.payload->size(), 1U);
        EXPECT_EQ(product.payload->query_revision, 7U);
    }
}