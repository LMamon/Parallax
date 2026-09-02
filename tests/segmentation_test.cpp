#include <parallax/core/product.hpp>
#include <parallax/perception/segmentation.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace {

    using parallax::core::ProductId;
    using parallax::core::ProductMetadata;
    using parallax::core::SourceId;
    using parallax::core::SourceObservation;

    using parallax::perception::ImageSpace;
    using parallax::perception::MaskLayout;
    using parallax::perception::MaskRepresentation;
    using parallax::perception::SegmentationMask;

    [[nodiscard]] SegmentationMask make_valid_mask() {
        SegmentationMask mask{};

        mask.source_observation = SourceObservation{SourceId::StereoCamera, 42};
        mask.image_space = ImageSpace::RgbLeft;
        mask.query_revision = 7;

        mask.width = 1920;
        mask.height = 1200;
        mask.pitch_bytes = 1920;

        mask.layout = MaskLayout::RowMajor;
        mask.representation = MaskRepresentation::CudaDevice;

        mask.confidence = 0.93F;
        mask.mask_valid = true;

        mask.storage = std::make_shared<std::uint8_t>(1);
        return mask;
    }

    TEST(SegmentationMaskTest, ValidContractDoesNotRequireHostMask) {
        const auto mask = make_valid_mask();

        EXPECT_TRUE(mask.valid());
        EXPECT_EQ(mask.representation, MaskRepresentation::CudaDevice);
    }

    TEST(SegmentationMaskTest, UnknownImageSpaceIsInvalid) {
        auto mask = make_valid_mask();
        mask.image_space = ImageSpace::Unknown;

        EXPECT_FALSE(mask.valid());
    }

    TEST(SegmentationMaskTest, MissingSourceObservationIsInvalid) {
        auto mask = make_valid_mask();
        mask.source_observation = {};

        EXPECT_FALSE(mask.valid());
    }

    TEST(SegmentationMaskTest, MissingQueryRevisionIsInvalid) {
        auto mask = make_valid_mask();
        mask.query_revision = 0;

        EXPECT_FALSE(mask.valid());
    }

    TEST(SegmentationMaskTest, MissingStorageIsInvalid) {
        auto mask = make_valid_mask();
        mask.storage.reset();

        EXPECT_FALSE(mask.valid());
    }

    TEST(SegmentationMaskTest, ProductPreservesCameraObservation) {
        ProductMetadata metadata{};
        metadata.observation =SourceObservation{SourceId::StereoCamera, 42};
        metadata.valid = true;

        std::shared_ptr<const SegmentationMask> payload = std::make_shared<SegmentationMask>(make_valid_mask());

        const auto product = parallax::core::make_product<SegmentationMask>(ProductId::Segmentation,
                                                                            metadata,
                                                                            std::move(payload));

        ASSERT_TRUE(product.valid());
        ASSERT_TRUE(product.payload->valid());

        EXPECT_EQ(product.metadata.observation, product.payload->source_observation);
        EXPECT_EQ(product.payload->query_revision, 7U);
        EXPECT_EQ(product.payload->image_space, ImageSpace::RgbLeft);
    }
}