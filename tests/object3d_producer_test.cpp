#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/object3d.hpp>
#include <parallax/perception/object3d_producer.hpp>
#include <parallax/perception/stereo_roi_associator.hpp>
#include <parallax/stereo/calibration.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

    using namespace parallax;
    using Clock = std::chrono::steady_clock;

    constexpr std::uint32_t Width = 9;
    constexpr std::uint32_t Height = 9;

    std::vector<float> identity_map_x() {
        std::vector<float> map;
        map.reserve(Width * Height);

        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                map.push_back(static_cast<float>(x));
            }
        }
        return map;
    }

    std::vector<float> identity_map_y() {
        std::vector<float> map;
        map.reserve(Width * Height);

        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                map.push_back(static_cast<float>(y));
            }
        }

        return map;
    }

    perception::RectifiedCameraModel camera_model() {
        perception::RectifiedCameraModel model{};
        model.fx_px = 100.0F;
        model.fy_px = 100.0F;
        model.cx_px = 4.0F;
        model.cy_px = 4.0F;
        model.coordinate_frame = "camera_left_optical";
        return model;
    }

    core::ProductMetadata metadata(std::uint64_t sequence, Clock::time_point timestamp) {

        core::ProductMetadata result{};

        result.observation = {core::SourceId::StereoCamera, sequence};

        result.timestamp = timestamp;
        result.production_timestamp = timestamp;
        result.valid = true;

        return result;
    }

    std::shared_ptr<const perception::DetectionSet> detection_payload() {

        auto detections = std::make_shared<perception::DetectionSet>();

        detections->query = "cup";
        detections->query_revision = 3;
        detections->image_space = perception::ImageSpace::RgbLeft;

        detections->boxes.push_back({3.0F, 3.0F, 2.0F, 2.0F});

        detections->scores.push_back(0.9F);
        detections->labels.push_back(0);

        return detections;
    }

    core::Product<isp::DepthFrame> make_depth(core::ExecutionContext& context,
                                              float depth_m,
                                              std::uint64_t sequence,
                                              Clock::time_point timestamp) {

        auto payload = std::make_shared<isp::DepthFrame>();

        EXPECT_TRUE(payload->depth.allocate(Width, Height, 1, sizeof(float)));

        payload->width = Width;
        payload->height = Height;

        std::vector<float> values(Width * Height, depth_m);

        auto& lane = context.stereoLane();
        EXPECT_TRUE(payload->depth.uploadAsync(values.data(), Width * sizeof(float), lane.cudaHandle()));

        auto completion = context.recordCudaCompletion(lane.cudaHandle());
        EXPECT_TRUE(completion.valid());

        std::shared_ptr<const isp::DepthFrame> const_payload = payload;

        return core::make_product<isp::DepthFrame>(core::ProductId::Depth,
                                                   metadata(sequence, timestamp),
                                                   std::move(const_payload),
                                                   std::move(completion));
    }

    void initialize_associator(perception::StereoRoiAssociator& associator) {
        ASSERT_TRUE(associator.initialize(Width, Height, identity_map_x(), identity_map_y(), camera_model()));
    }
}

TEST(Object3DProducerTest, PublishesExactObservationAssociation) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    auto& store = context.products();

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "camera_left_optical"};

    initialize_associator(associator);
    perception::Object3DProducer producer{associator, store};

    const auto now = Clock::now();
    store.publish(core::make_product<perception::DetectionSet>(core::ProductId::Detection,
                                                               metadata(10, now), 
                                                               detection_payload()));

    store.publish(make_depth(context, 2.0F, 10, now));
    ASSERT_EQ(producer.submit(context), core::SubmitResult::Submitted);

    const auto output = store.latest<perception::Object3DSet>(core::ProductId::Object3D);

    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(output->valid());
    ASSERT_EQ(output->payload->size(), 1U);

    const auto& object = output->payload->objects.front();

    EXPECT_EQ(output->metadata.observation.sequence, 10U);
    EXPECT_EQ(object.semantic_observation.sequence, 10U);
    EXPECT_EQ(object.metric_observation.sequence, 10U);
    EXPECT_FLOAT_EQ(object.depth_m, 2.0F);

    context.shutdown();
}

TEST(Object3DProducerTest, UsesBoundedNearestSameSourceDepth) {

    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    auto& store = context.products();
    store.set_history_capacity(core::ProductId::Depth, 4);

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "camera_left_optical"};

    initialize_associator(associator);
    perception::Object3DProducer producer{associator, store};

    const auto now = Clock::now();

    store.publish(core::make_product<perception::DetectionSet>(core::ProductId::Detection,
                                                               metadata(20, now), 
                                                               detection_payload()));

    store.publish(make_depth(context, 2.5F, 19, now - std::chrono::milliseconds{20}));

    ASSERT_EQ(producer.submit(context), core::SubmitResult::Submitted);

    const auto output = store.latest<perception::Object3DSet>(core::ProductId::Object3D);

    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->payload->size(), 1U);

    const auto& object = output->payload->objects.front();

    EXPECT_EQ(object.semantic_observation.sequence, 20U);
    EXPECT_EQ(object.metric_observation.sequence, 19U);
    EXPECT_EQ(object.source_time_delta, std::chrono::milliseconds{20});

    EXPECT_FLOAT_EQ(object.depth_m, 2.5F);
    context.shutdown();
}

TEST(Object3DProducerTest, RejectsDepthOutsideCompatibilityBound) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    auto& store = context.products();
    store.set_history_capacity(core::ProductId::Depth, 4);

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "test_left_camera"};

    initialize_associator(associator);

    perception::Object3DProducer producer{associator, store};

    const auto now = Clock::now();

    store.publish(core::make_product<perception::DetectionSet>(core::ProductId::Detection,
                                                               metadata(30, now),
                                                               detection_payload()));
    store.publish(make_depth(context, 2.0F, 29, now - std::chrono::milliseconds{100}));

    EXPECT_EQ(producer.submit(context), core::SubmitResult::NoWork);
    EXPECT_EQ(store.latest<perception::Object3DSet>(core::ProductId::Object3D), nullptr);

    context.shutdown();
}

TEST(Object3DProducerTest, PublishesValidEmptySetWhenDetectionHasNoObjects) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    auto& store = context.products();

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "test_left_camera"};

    initialize_associator(associator);

    perception::Object3DProducer producer{associator, store};

    auto empty_detection = std::make_shared<perception::DetectionSet>();

    empty_detection->query = "cup";
    empty_detection->query_revision = 4;
    empty_detection->image_space = perception::ImageSpace::RgbLeft;

    const auto now = Clock::now();
    std::shared_ptr<const perception::DetectionSet> const_detection = empty_detection;

    store.publish(core::make_product<perception::DetectionSet>(core::ProductId::Detection,
                                                                metadata(40, now),
                                                                std::move(const_detection)));

    store.publish(make_depth(context, 2.0F, 40, now));

    ASSERT_EQ(producer.submit(context), core::SubmitResult::Submitted);

    const auto output = store.latest<perception::Object3DSet>(core::ProductId::Object3D);

    ASSERT_NE(output, nullptr);
    EXPECT_TRUE(output->payload->valid());
    EXPECT_TRUE(output->payload->empty());

    context.shutdown();
}