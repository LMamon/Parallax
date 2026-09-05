#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/isp/frame_types.hpp>
#include <parallax/pose/charuco_pose.hpp>
#include <parallax/pose/marker_depth_producer.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {
    using namespace parallax;
    using Clock = std::chrono::steady_clock;

    constexpr std::uint32_t Width = 9;
    constexpr std::uint32_t Height = 9;

    core::ProductMetadata make_metadata(std::uint64_t sequence, Clock::time_point timestamp) {
        core::ProductMetadata metadata{};

        metadata.observation = {core::SourceId::StereoCamera, sequence};

        metadata.timestamp = timestamp;
        metadata.production_timestamp = timestamp;
        metadata.valid = true;

        return metadata;
    }

    std::shared_ptr<pose::CharucoPoseResult> make_pose_result() {
        auto result = std::make_shared<pose::CharucoPoseResult>();

        result->pose_valid = true;
        result->plane_valid = true;
        result->projected_center = {4.0F, 4.0F};

        return result;
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

    void publish_pose(core::ProductStore& store, std::uint64_t sequence, Clock::time_point timestamp) {
        std::shared_ptr<const pose::CharucoPoseResult> payload = make_pose_result();

        store.publish(core::make_product<pose::CharucoPoseResult>(core::ProductId::Pose,
                                                                  make_metadata(sequence, timestamp),
                                                                  std::move(payload)));
    }
}

TEST(MarkerDepthProducerTest, UsesNeighborhoodMedianForMarkerDepth) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    core::ProductStore store;
    pose::MarkerDepthPoducer producer{store};

    const auto now = Clock::now();
    publish_pose(store, 10, now);

    std::vector<float> values(Width * Height, std::numeric_limits<float>::quiet_NaN());

    // Seven valid samples with one large outlier. Median is 4 m.
    values[3 * Width + 3] = 1.0F;
    values[3 * Width + 4] = 2.0F;
    values[3 * Width + 5] = 3.0F;
    values[4 * Width + 3] = 4.0F;
    values[4 * Width + 4] = 5.0F;
    values[4 * Width + 5] = 6.0F;
    values[5 * Width + 4] = 100.0F;

    store.publish(make_depth_product(context, values, 10, now));

    ASSERT_EQ(producer.submit(context), core::SubmitResult::Submitted);

    const auto marker_depth = store.latest<pose::CharucoPoseResult>(core::ProductId::MarkerDepth);

    ASSERT_NE(marker_depth, nullptr);
    ASSERT_TRUE(marker_depth->valid());

    EXPECT_TRUE(marker_depth->payload->depth_valid);
    EXPECT_FLOAT_EQ(marker_depth->payload->depth_m, 4.0F);

    context.shutdown();
}

TEST(MarkerDepthProducerTest, ToleratesInvalidCenterWithSupportedNeighbors) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    core::ProductStore store;
    pose::MarkerDepthPoducer producer{store};

    const auto now = Clock::now();
    publish_pose(store, 20, now);

    std::vector<float> values(Width * Height, 2.0F);

    values[4 * Width + 4] = std::numeric_limits<float>::quiet_NaN();
    store.publish(make_depth_product(context, values, 20, now));

    ASSERT_EQ(producer.submit(context), core::SubmitResult::Submitted);

    const auto marker_depth = store.latest<pose::CharucoPoseResult>(core::ProductId::MarkerDepth);

    ASSERT_NE(marker_depth, nullptr);

    EXPECT_TRUE(marker_depth->payload->depth_valid);
    EXPECT_FLOAT_EQ(marker_depth->payload->depth_m, 2.0F);

    context.shutdown();
}

TEST(MarkerDepthProducerTest, PreservesProjectionWhenDepthSupportIsInsufficient) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    core::ProductStore store;
    pose::MarkerDepthPoducer producer{store};

    const auto now = Clock::now();

    publish_pose(store, 30, now);

    std::vector<float> values(Width * Height, std::numeric_limits<float>::quiet_NaN());

    // Four valid samples remain below MinValidSamples.
    values[3 * Width + 3] = 2.0F;
    values[3 * Width + 4] = 2.0F;
    values[4 * Width + 3] = 2.0F;
    values[4 * Width + 4] = 2.0F;

    store.publish(make_depth_product(context, values, 30, now));

    ASSERT_EQ(producer.submit(context), core::SubmitResult::Submitted);

    const auto projection = store.latest<pose::CharucoPoseResult>(core::ProductId::Projection);
    const auto marker_depth = store.latest<pose::CharucoPoseResult>(core::ProductId::MarkerDepth);

    ASSERT_NE(projection, nullptr);
    ASSERT_NE(marker_depth, nullptr);

    EXPECT_FALSE(marker_depth->payload->depth_valid);
    EXPECT_FLOAT_EQ(marker_depth->payload->depth_m, 0.0F);

    EXPECT_TRUE(projection->payload->pose_valid);
    EXPECT_TRUE(projection->payload->plane_valid);

    context.shutdown();
}

TEST(MarkerDepthProducerTest, RejectsDifferentDepthObservation) {
    core::ExecutionContext context;
    ASSERT_TRUE(context.initialize());

    core::ProductStore store;
    pose::MarkerDepthPoducer producer{store};

    const auto now = Clock::now();
    publish_pose(store, 40, now);

    std::vector<float> values(Width * Height, 2.0F);

    store.publish(make_depth_product(context, values, 41, now));

    EXPECT_EQ(producer.submit(context), core::SubmitResult::NoWork);
    EXPECT_EQ(store.latest<pose::CharucoPoseResult>(core::ProductId::MarkerDepth), nullptr);

    context.shutdown();
}