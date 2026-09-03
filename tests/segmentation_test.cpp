#include <parallax/core/product.hpp>
#include <parallax/perception/segmentation.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/segmentation_compatibility.hpp>


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

    parallax::core::Product<parallax::isp::StereoRgbFrame>
        make_rgb_product(std::uint64_t sequence) {
            parallax::core::ProductMetadata metadata{};

            metadata.observation = {parallax::core::SourceId::StereoCamera, sequence};
            metadata.valid = true;

            auto frame = std::make_shared<parallax::isp::StereoRgbFrame>();

            frame->width = 1920;
            frame->height = 1200;

            return parallax::core::make_product<parallax::isp::StereoRgbFrame>(
                                                parallax::core::ProductId::RgbLeft,
                                                metadata,
                                                std::move(frame));
        }


        parallax::core::Product<parallax::perception::DetectionSet>
        make_detection_product(std::uint64_t sequence) {
            parallax::core::ProductMetadata metadata{};

            metadata.observation = {parallax::core::SourceId::StereoCamera, sequence};

            metadata.valid = true;

            auto detection = std::make_shared<parallax::perception::DetectionSet>();

            detection->query = "person";
            detection->query_revision = 1;
            detection->image_space = parallax::perception::ImageSpace::RgbLeft;

            detection->boxes.emplace_back(10.0F, 20.0F, 100.0F, 120.0F);

            detection->scores.push_back(0.9F);
            detection->labels.push_back(0);

            return parallax::core::make_product<parallax::perception::DetectionSet>(
                                                parallax::core::ProductId::Detection,
                                                metadata,
                                                std::move(detection));
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

    TEST(SegmentationCompatibilityTest, UsesExactCurrentRgbObservation) {
        parallax::core::ProductStore store;

        store.publish(make_rgb_product(100));

        const auto detection = make_detection_product(100);
        const auto rgb = parallax::perception::find_segmentation_rgb(store, detection);

        ASSERT_NE(rgb, nullptr);
        EXPECT_EQ(rgb->metadata.observation.sequence, 100U);
    }


    TEST(SegmentationCompatibilityTest, UsesRetainedRgbInsteadOfLatestRgb) {
        parallax::core::ProductStore store;

        store.set_history_capacity(parallax::core::ProductId::RgbLeft, 2);

        store.publish(make_rgb_product(120));
        store.publish(make_rgb_product(121));

        const auto detection = make_detection_product(120);
        const auto rgb = parallax::perception::find_segmentation_rgb(store, detection);

        ASSERT_NE(rgb, nullptr);
        EXPECT_EQ(rgb->metadata.observation.sequence, 120U);

        const auto latest = store.latest<parallax::isp::StereoRgbFrame>(parallax::core::ProductId::RgbLeft);

        ASSERT_NE(latest, nullptr);
        EXPECT_EQ(latest->metadata.observation.sequence, 121U);
    }


    TEST(SegmentationCompatibilityTest, RejectsLatestRgbWhenDetectionSourceWasEvicted) {
        parallax::core::ProductStore store;

        store.set_history_capacity(parallax::core::ProductId::RgbLeft, 2);

        store.publish(make_rgb_product(200));
        store.publish(make_rgb_product(201));
        store.publish(make_rgb_product(202));

        const auto detection = make_detection_product(200);
        const auto rgb = parallax::perception::find_segmentation_rgb(store, detection);

        EXPECT_EQ(rgb, nullptr);

        const auto latest = store.latest<parallax::isp::StereoRgbFrame>(parallax::core::ProductId::RgbLeft);

        ASSERT_NE(latest, nullptr);
        EXPECT_EQ(latest->metadata.observation.sequence, 202U);
    }


    TEST(SegmentationCompatibilityTest, RejectsWrongImageSpace) {
        parallax::core::ProductStore store;
        store.publish(make_rgb_product(300));

        auto detection = make_detection_product(300);
        auto mutable_detection = std::make_shared<parallax::perception::DetectionSet>(*detection.payload);

        mutable_detection->image_space = parallax::perception::ImageSpace::RectifiedLeft;
        detection.payload = std::move(mutable_detection);

        const auto rgb = parallax::perception::find_segmentation_rgb(store, detection);

        EXPECT_EQ(rgb, nullptr);
    }

    TEST(SegmentationCompatibilityTest, RgbRetentionRemainsBoundedAcrossManyFrames) {
        parallax::core::ProductStore store;
        store.set_history_capacity(parallax::core::ProductId::RgbLeft, 2);

        for (std::uint64_t sequence = 1; sequence <= 100; ++sequence) {
            store.publish(make_rgb_product(sequence));
        }

        EXPECT_EQ(store.history_capacity(parallax::core::ProductId::RgbLeft), 2U);
        EXPECT_EQ(parallax::perception::find_segmentation_rgb(store, make_detection_product(1)), nullptr);
        ASSERT_NE(parallax::perception::find_segmentation_rgb(store, make_detection_product(99)), nullptr);
        ASSERT_NE(parallax::perception::find_segmentation_rgb(store, make_detection_product(100)), nullptr);
    }
}