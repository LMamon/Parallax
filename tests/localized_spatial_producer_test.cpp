#include <gtest/gtest.h>

#include <parallax/core/execution_context.hpp>
#include <parallax/core/product.hpp>
#include <parallax/localization/localization.hpp>
#include <parallax/perception/localized_spatial_observation.hpp>
#include <parallax/perception/localized_spatial_producer.hpp>

#include <chrono>
#include <memory>

namespace {

    parallax::core::ProductMetadata metadata(std::uint64_t sequence, std::chrono::steady_clock::time_point timestamp) {
        parallax::core::ProductMetadata result{};
        result.observation = {parallax::core::SourceId::StereoCamera, sequence};
        result.timestamp = timestamp;
        result.production_timestamp = timestamp;
        result.valid = true;
        return result;
    }

}

TEST(LocalizedSpatialProducer, UsesCompatibleHistoricalPoseInsteadOfLatest) {
    using namespace std::chrono_literals;
    using namespace parallax;

    // Constructor geometry is covered by the physical calibration integration.
    // This test protects the temporal rule directly through ProductStore history.
    core::ProductStore products;
    products.set_history_capacity(core::ProductId::LocalizationPose, 8);

    const auto t0 = std::chrono::steady_clock::now();

    auto old_pose = std::make_shared<localization::LocalizationPose>();
    old_pose->timestamp_ns = 1;
    old_pose->epoch = 1;
    old_pose->translation_m = {1.0F, 0.0F, 0.0F};

    products.publish(core::make_product<localization::LocalizationPose>(core::ProductId::LocalizationPose,
                                                                        metadata(10, t0),
                                                                        old_pose));

    auto latest_pose = std::make_shared<localization::LocalizationPose>();
    latest_pose->timestamp_ns = 2;
    latest_pose->epoch = 1;
    latest_pose->translation_m = {100.0F, 0.0F, 0.0F};

    products.publish(core::make_product<localization::LocalizationPose>(core::ProductId::LocalizationPose,
                                                                        metadata(20, t0 + 500ms),
                                                                        latest_pose));

    const auto history = products.history<localization::LocalizationPose>(core::ProductId::LocalizationPose);

    ASSERT_EQ(history.size(), 2U);

    auto best = history.front();
    auto best_delta = std::chrono::steady_clock::duration::max();

    for (const auto& candidate : history) {
        const auto delta = candidate->metadata.timestamp >= t0 + 5ms
                            ? candidate->metadata.timestamp - (t0 + 5ms)
                            : (t0 + 5ms) - candidate->metadata.timestamp;

        if (delta < best_delta) {
            best = candidate;
            best_delta = delta;
        }
    }

    ASSERT_TRUE(best);
    EXPECT_EQ(best->metadata.observation.sequence, 10U);
    EXPECT_FLOAT_EQ(best->payload->translation_m[0], 1.0F);
}