#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/lidar/frame_types.hpp>
#include <parallax/perception/detection.hpp>
#include <parallax/perception/lidar_detection_associator.hpp>
#include <parallax/perception/object3d.hpp>
#include <parallax/perception/object3d_producer.hpp>
#include <parallax/perception/stereo_roi_associator.hpp>
#include <parallax/stereo/calibration.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <utility>

namespace {
    using namespace parallax;
    using Clock = std::chrono::steady_clock;

    constexpr std::uint32_t Width = 101;
    constexpr std::uint32_t Height = 101;

    std::vector<float> identityMapX() {
        std::vector<float> map;
        map.reserve(Width * Height);

        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                map.push_back(static_cast<float>(x));
            }
        }
        return map;
    }

    std::vector<float> identityMapY() {
        std::vector<float> map;
        map.reserve(Width * Height);

        for (std::uint32_t y = 0; y < Height; ++y) {
            for (std::uint32_t x = 0; x < Width; ++x) {
                map.push_back(static_cast<float>(y));
            }
        }
        return map;
    }

    core::ProductMetadata cameraMetadata(std::uint64_t sequence,
                                         Clock::time_point timestamp) {
        core::ProductMetadata metadata{};
        metadata.observation = {core::SourceId::StereoCamera, sequence};
        metadata.timestamp = timestamp;
        metadata.production_timestamp = timestamp;
        metadata.valid = true;
        return metadata;
    }

    core::ProductMetadata lidarMetadata(std::uint64_t sequence,
                                        Clock::time_point timestamp) {
        core::ProductMetadata metadata{};
        metadata.observation = {core::SourceId::Rplidar, sequence};
        metadata.timestamp = timestamp;
        metadata.production_timestamp = timestamp;
        metadata.valid = true;
        return metadata;
    }

    std::shared_ptr<const perception::DetectionSet> detectionPayload() {
        auto detections = std::make_shared<perception::DetectionSet>();
        detections->query = "cup";
        detections->query_revision = 1;
        detections->image_space = perception::ImageSpace::RgbLeft;
        detections->boxes.push_back({45.0F, 45.0F, 10.0F, 10.0F});
        detections->scores.push_back(0.95F);
        detections->labels.push_back(0);
        return detections;
    }

    void initializeLidarAssociator(perception::LidarDetectionAssociator& associator) {
        // Synthetic camera: lidar +X maps to camera +Z and lidar +Y maps
        // to camera +X. This makes a zero-degree 2 m hit land at (50, 50).
        const std::array<double, 9> rectified_from_lidar{
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0
        };

        ASSERT_TRUE(associator.initialize(
            Width,
            Height,
            50.0F,
            50.0F,
            50.0F,
            50.0F,
            rectified_from_lidar,
            {0.0, 0.0, 0.0},
            identityMapX(),
            identityMapY(),
            "camera_left_optical"));
    }

    core::Product<lidar::LidarScan> lidarProduct(
        std::vector<lidar::LidarPoint> points,
        std::uint64_t sequence,
        Clock::time_point timestamp) {

        auto payload = std::make_shared<lidar::LidarScan>();
        payload->points = std::move(points);
        std::shared_ptr<const lidar::LidarScan> const_payload = payload;

        return core::make_product<lidar::LidarScan>(
            core::ProductId::LidarScan,
            lidarMetadata(sequence, timestamp),
            std::move(const_payload));
    }

    perception::RectifiedCameraModel stereoCameraModel() {
        perception::RectifiedCameraModel model{};
        model.fx_px = 50.0F;
        model.fy_px = 50.0F;
        model.cx_px = 50.0F;
        model.cy_px = 50.0F;
        model.coordinate_frame = "camera_left_optical";
        return model;
    }

    core::Product<isp::DepthFrame> depthProduct(
        core::ExecutionContext& context,
        float depth_m,
        std::uint64_t sequence,
        Clock::time_point timestamp) {

        auto payload = std::make_shared<isp::DepthFrame>();
        EXPECT_TRUE(payload->depth.allocate(Width, Height, 1, sizeof(float)));
        payload->width = Width;
        payload->height = Height;

        std::vector<float> values(Width * Height, depth_m);
        auto& lane = context.stereoLane();
        EXPECT_TRUE(payload->depth.uploadAsync(
            values.data(),
            Width * sizeof(float),
            lane.cudaHandle()));

        auto completion = context.recordCudaCompletion(lane.cudaHandle());
        EXPECT_TRUE(completion.valid());

        std::shared_ptr<const isp::DepthFrame> const_payload = payload;
        return core::make_product<isp::DepthFrame>(
            core::ProductId::Depth,
            cameraMetadata(sequence, timestamp),
            std::move(const_payload),
            std::move(completion));
    }
}

TEST(LidarDetectionAssociatorTest, CenterHitCreatesLidarObject3D) {
    perception::LidarDetectionAssociator associator;
    initializeLidarAssociator(associator);

    const auto now = Clock::now();
    const auto detections = detectionPayload();

    const auto scan = lidarProduct(
        {{0.0F, 2.0F, 20, true}},
        7,
        now);

    perception::Object3DSet output{};
    ASSERT_TRUE(associator.associate(
        *detections,
        cameraMetadata(10, now),
        scan,
        output));

    ASSERT_EQ(output.size(), 1U);
    const auto& object = output.objects.front();
    EXPECT_EQ(object.method, perception::Object3DMethod::LidarAssociation);
    EXPECT_EQ(object.geometry, perception::Object3DGeometry::Point);
    EXPECT_EQ(object.metric_observation.source, core::SourceId::Rplidar);
    EXPECT_FLOAT_EQ(object.range_m, 2.0F);
    ASSERT_TRUE(object.lidar_evidence.has_value());
    EXPECT_TRUE(object.lidar_evidence->valid());
    EXPECT_EQ(object.lidar_evidence->observation.sequence, 7U);
    EXPECT_FLOAT_EQ(object.lidar_evidence->range_m, 2.0F);
    EXPECT_NEAR(object.position_m[0], 0.0F, 1.0e-5F);
    EXPECT_NEAR(object.position_m[1], 0.0F, 1.0e-5F);
    EXPECT_NEAR(object.position_m[2], 2.0F, 1.0e-5F);
}

TEST(LidarDetectionAssociatorTest, PositiveNativeAngleProjectsToPositiveImageSide) {
    perception::LidarDetectionAssociator associator;
    initializeLidarAssociator(associator);

    perception::DetectionSet detections{};
    detections.query = "cup";
    detections.query_revision = 1;
    detections.image_space = perception::ImageSpace::RgbLeft;
    detections.boxes.push_back({53.0F, 45.0F, 5.0F, 10.0F});
    detections.scores.push_back(0.95F);
    detections.labels.push_back(0);

    const auto now = Clock::now();
    const auto scan = lidarProduct(
        {{0.10F, 2.0F, 20, true}},
        8,
        now);

    perception::Object3DSet output{};
    ASSERT_TRUE(associator.associate(
        detections,
        cameraMetadata(11, now),
        scan,
        output));

    ASSERT_EQ(output.size(), 1U);
    EXPECT_EQ(output.objects.front().method, perception::Object3DMethod::LidarAssociation);
}

TEST(LidarDetectionAssociatorTest, ReturnOutsideDetectionDoesNotAssociate) {
    perception::LidarDetectionAssociator associator;
    initializeLidarAssociator(associator);

    const auto now = Clock::now();
    const auto detections = detectionPayload();

    const auto scan = lidarProduct(
        {{0.30F, 2.0F, 20, true}},
        8,
        now);

    perception::Object3DSet output{};
    ASSERT_TRUE(associator.associate(
        *detections,
        cameraMetadata(11, now),
        scan,
        output));

    EXPECT_TRUE(output.empty());
}

TEST(LidarFirstObject3DTest, LidarHitOverridesAvailableStereoDepth) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());
    auto& store = context.products();

    store.set_history_capacity(core::ProductId::Depth, 4);
    store.set_history_capacity(core::ProductId::LidarScan, 4);

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator stereo_associator{calibration, "camera_left_optical"};
    ASSERT_TRUE(stereo_associator.initialize(
        Width,
        Height,
        identityMapX(),
        identityMapY(),
        stereoCameraModel()));

    perception::LidarDetectionAssociator lidar_associator;
    initializeLidarAssociator(lidar_associator);

    perception::Object3DProducer producer{
        stereo_associator,
        lidar_associator,
        store};

    const auto now = Clock::now();

    store.publish(core::make_product<perception::DetectionSet>(
        core::ProductId::Detection,
        cameraMetadata(20, now),
        detectionPayload()));

    // Stereo says 4 m.
    store.publish(depthProduct(context, 4.0F, 20, now));

    // LiDAR intersects the detection at 2 m.
    store.publish(lidarProduct(
        {{0.0F, 2.0F, 20, true}},
        50,
        now - std::chrono::milliseconds{10}));

    ASSERT_EQ(producer.submit(context), core::SubmitResult::Submitted);

    const auto output =
        store.latest<perception::Object3DSet>(core::ProductId::Object3D);

    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(output->valid());
    ASSERT_EQ(output->payload->size(), 1U);

    const auto& object = output->payload->objects.front();
    EXPECT_EQ(object.method, perception::Object3DMethod::StereoLidarRefined);
    EXPECT_EQ(object.metric_observation.source, core::SourceId::Rplidar);
    EXPECT_EQ(object.metric_observation.sequence, 50U);
    EXPECT_FLOAT_EQ(object.range_m, 2.0F);
    EXPECT_NEAR(object.position_m[2], 2.0F, 1.0e-5F);
    EXPECT_NE(object.depth_m, 4.0F);

    ASSERT_TRUE(object.stereo_evidence.has_value());
    ASSERT_TRUE(object.lidar_evidence.has_value());
    EXPECT_TRUE(object.stereo_evidence->valid());
    EXPECT_TRUE(object.lidar_evidence->valid());

    EXPECT_EQ(object.stereo_evidence->observation.source, core::SourceId::StereoCamera);
    EXPECT_EQ(object.stereo_evidence->observation.sequence, 20U);
    EXPECT_FLOAT_EQ(object.stereo_evidence->depth_m, 4.0F);

    EXPECT_EQ(object.lidar_evidence->observation.source, core::SourceId::Rplidar);
    EXPECT_EQ(object.lidar_evidence->observation.sequence, 50U);
    EXPECT_FLOAT_EQ(object.lidar_evidence->range_m, 2.0F);
    EXPECT_NEAR(object.lidar_evidence->position_m[2], 2.0F, 1.0e-5F);

    context.shutdown();
}
