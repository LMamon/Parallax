#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/perception/stereo_roi_associator.hpp>
#include <parallax/stereo/calibration.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
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

    perception::RectifiedCameraModel test_camera_model() {
        perception::RectifiedCameraModel model{};

        model.fx_px = 100.0F;
        model.fy_px = 100.0F;
        model.cx_px = 4.0F;
        model.cy_px = 4.0F;
        model.coordinate_frame = "camera_left_optical";

        return model;
    }

    perception::DetectionSet make_detection(const cv::Rect2f& box, float score = 0.9F) {
        perception::DetectionSet detections{};

        detections.query = "cup";
        detections.query_revision = 7;
        detections.image_space = perception::ImageSpace::RgbLeft;

        detections.boxes.push_back(box);
        detections.scores.push_back(score);
        detections.labels.push_back(0);

        return detections;
    }

    core::ProductMetadata make_metadata(std::uint64_t sequence, Clock::time_point timestamp) {
        core::ProductMetadata metadata{};

        metadata.observation = {core::SourceId::StereoCamera, sequence};

        metadata.timestamp = timestamp;
        metadata.production_timestamp = timestamp;
        metadata.valid = true;

        return metadata;
    }

    core::Product<isp::DepthFrame> make_depth_product(core::ExecutionContext& context,
                                                      const std::vector<float>& values,
                                                      std::uint64_t sequence,
                                                      Clock::time_point timestamp) {

        auto payload = std::make_shared<isp::DepthFrame>();
        EXPECT_TRUE(payload->depth.allocate(Width, Height, 1, sizeof(float)));

        payload->width = Width;
        payload->height = Height;

        auto& lane = context.stereoLane();
        EXPECT_TRUE(payload->depth.uploadAsync(values.data(), Width * sizeof(float), lane.cudaHandle()));
        auto completion = context.recordCudaCompletion(lane.cudaHandle());
        EXPECT_TRUE(completion.valid());

        std::shared_ptr<const isp::DepthFrame> const_payload = payload;

        return core::make_product<isp::DepthFrame>(core::ProductId::Depth,
                                                   make_metadata(sequence, timestamp),
                                                   std::move(const_payload),
                                                   std::move(completion));
    }

    void initialize_associator(perception::StereoRoiAssociator& associator) {
        ASSERT_TRUE(associator.initialize(Width, Height, identity_map_x(), identity_map_y(), test_camera_model()));
    }
}

TEST(StereoRoiAssociatorTest, ProducesObject3DFromCudaResidentDepthMedian) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "camera_left_optical"};
    initialize_associator(associator);

    std::vector<float> depth_values(Width * Height, 2.0F);

    const auto now = Clock::now();
    auto depth = make_depth_product(context, depth_values, 10, now);
    const auto detections = make_detection({3.0F, 3.0F, 2.0F, 2.0F});

    perception::Object3DSet output{};

    ASSERT_TRUE(associator.associate(detections, make_metadata(10, now), depth, context, output));
    ASSERT_TRUE(output.valid());
    ASSERT_EQ(output.size(), 1U);

    const auto& object = output.objects.front();

    EXPECT_EQ(object.label, "cup");
    EXPECT_FLOAT_EQ(object.depth_m, 2.0F);

    EXPECT_NEAR(object.position_m[0], 0.0F, 1e-6F);
    EXPECT_NEAR(object.position_m[1], 0.0F, 1e-6F);
    EXPECT_NEAR(object.position_m[2], 2.0F, 1e-6F);

    EXPECT_EQ(object.method, perception::Object3DMethod::StereoRoi);
    EXPECT_EQ(object.geometry, perception::Object3DGeometry::Point);
    EXPECT_FALSE(object.persistent());

    context.shutdown();
}

TEST(StereoRoiAssociatorTest, BackProjectsUsingRectifiedCameraModel) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "camera_left_optical"};
    initialize_associator(associator);

    std::vector<float> depth_values(Width * Height, 2.0F);

    const auto now = Clock::now();
    auto depth = make_depth_product(context, depth_values, 20, now);

    /*
     * Box center is (5, 4). With fx=100, cx=4 and Z=2:
     *
     * X = (5 - 4) * 2 / 100 = 0.02 m
     */
    const auto detections = make_detection({4.0F, 3.0F, 2.0F, 2.0F});

    perception::Object3DSet output{};

    ASSERT_TRUE(associator.associate(detections, make_metadata(20, now), depth, context, output));
    ASSERT_EQ(output.size(), 1U);
    EXPECT_NEAR(output.objects[0].position_m[0], 0.02F, 1e-6F);
    EXPECT_NEAR(output.objects[0].position_m[1], 0.0F, 1e-6F);
    EXPECT_NEAR(output.objects[0].position_m[2], 2.0F, 1e-6F);

    context.shutdown();
}

TEST(StereoRoiAssociatorTest, UsesValidNeighborsWhenCenterDepthIsInvalid) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "camera_left_optical"};
    initialize_associator(associator);

    std::vector<float> depth_values(Width * Height, 2.0F);

    // Detection center is (4, 4). A single invalid center pixel must not
    // invalidate an otherwise well-supported 7x7 stereo neighborhood.
    depth_values[4 * Width + 4] =std::numeric_limits<float>::quiet_NaN();

    const auto now = Clock::now();
    auto depth = make_depth_product(context, depth_values, 30, now);
    const auto detections = make_detection({3.0F, 3.0F, 2.0F, 2.0F});

    perception::Object3DSet output{};

    ASSERT_TRUE(associator.associate(detections, make_metadata(30, now), depth, context, output));
    ASSERT_EQ(output.size(), 1U);
    EXPECT_FLOAT_EQ(output.objects[0].depth_m, 2.0F);
    EXPECT_GT(output.objects[0].support_quality, 0.9F);
    context.shutdown();
}

TEST(StereoRoiAssociatorTest, RejectsObjectWithInsufficientDepthSupport) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "camera_left_optical"};
    initialize_associator(associator);

    std::vector<float> depth_values(Width * Height, std::numeric_limits<float>::quiet_NaN());

    // Four valid samples remain below MinValidSamples=5.
    depth_values[3 * Width + 3] = 1.0F;
    depth_values[3 * Width + 4] = 1.0F;
    depth_values[4 * Width + 3] = 1.0F;
    depth_values[4 * Width + 4] = 1.0F;

    const auto now = Clock::now();
    auto depth = make_depth_product(context, depth_values, 40, now);
    const auto detections = make_detection({3.0F, 3.0F, 2.0F, 2.0F});

    perception::Object3DSet output{};

    ASSERT_TRUE(associator.associate(detections, make_metadata(40, now), depth, context, output));

    EXPECT_TRUE(output.valid());
    EXPECT_TRUE(output.empty());

    context.shutdown();
}

TEST(StereoRoiAssociatorTest, UsesMedianRatherThanSingleDepthSample) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "camera_left_optical"};
    initialize_associator(associator);

    std::vector<float> depth_values(Width * Height, std::numeric_limits<float>::quiet_NaN());

    /*
     * Seven supported pixels contain one extreme outlier. Sorted values:
     *
     * 1, 2, 3, 4, 5, 6, 100
     *
     * median = 4 m
     */
    depth_values[3 * Width + 3] = 1.0F;
    depth_values[3 * Width + 4] = 2.0F;
    depth_values[3 * Width + 5] = 3.0F;
    depth_values[4 * Width + 3] = 4.0F;
    depth_values[4 * Width + 4] = 5.0F;
    depth_values[4 * Width + 5] = 6.0F;
    depth_values[5 * Width + 4] = 100.0F;

    const auto now = Clock::now();
    auto depth = make_depth_product(context, depth_values, 50, now);
    const auto detections = make_detection({3.0F, 3.0F, 2.0F, 2.0F});

    perception::Object3DSet output{};

    ASSERT_TRUE(associator.associate(detections, make_metadata(50, now), depth, context, output));
    ASSERT_EQ(output.size(), 1U);
    EXPECT_FLOAT_EQ(output.objects[0].depth_m, 4.0F);

    context.shutdown();
}

TEST(StereoRoiAssociatorTest, SkipsUnmappableDetectionWithoutInvalidatingSet) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    stereo::StereoCalibration calibration;
    perception::StereoRoiAssociator associator{calibration, "camera_left_optical"};
    initialize_associator(associator);

    std::vector<float> depth_values(Width * Height, 2.0F);

    const auto now = Clock::now();
    auto depth = make_depth_product(context, depth_values, 60, now);
    const auto detections = make_detection({20.0F, 20.0F, 2.0F, 2.0F});

    perception::Object3DSet output{};

    ASSERT_TRUE(associator.associate(detections, make_metadata(60, now), depth, context, output));
    EXPECT_TRUE(output.valid());
    EXPECT_TRUE(output.empty());

    context.shutdown();
}