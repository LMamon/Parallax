#include <parallax/perception/object3d_association.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>

namespace {

    using Clock = std::chrono::steady_clock;
    using namespace parallax;

    core::Product<int> make_metric_product(std::uint64_t sequence,
                                           Clock::time_point timestamp,
                                           core::SourceId source = core::SourceId::StereoCamera) {

        core::ProductMetadata metadata{};
        metadata.observation = {source, sequence};
        metadata.timestamp = timestamp;
        metadata.production_timestamp = Clock::now();
        metadata.valid = true;

        return core::make_product<int>(core::ProductId::Depth, metadata, std::make_shared<const int>(static_cast<int>(sequence)));
    }

    core::ProductMetadata make_semantic_metadata(
        std::uint64_t sequence,
        Clock::time_point timestamp,
        core::SourceId source = core::SourceId::StereoCamera) {

        core::ProductMetadata metadata{};
        metadata.observation = {source, sequence};
        metadata.timestamp = timestamp;
        metadata.production_timestamp = Clock::now();
        metadata.valid = true;
        return metadata;
    }

}

TEST(Object3DAssociationTest, PrefersExactObservation) {
    core::ProductStore store;
    const auto now = Clock::now();

    store.set_history_capacity(core::ProductId::Depth, 4);
    store.publish(make_metric_product(40, now - std::chrono::milliseconds{10}));
    store.publish(make_metric_product(41, now));

    const auto semantic = make_semantic_metadata(41, now);
    const auto match = perception::find_metric_observation<int>(store,
                                                                core::ProductId::Depth,
                                                                semantic,
                                                                {});

    ASSERT_TRUE(match.matched());
    ASSERT_NE(match.product, nullptr);

    EXPECT_EQ(match.method, perception::Object3DMatchMethod::ExactObservation);
    EXPECT_EQ(match.product->metadata.observation.sequence, 41U);
    EXPECT_EQ(match.source_delta, Clock::duration::zero());
}

TEST(Object3DAssociationTest, SelectsNearestCompatibleTimestamp) {
    core::ProductStore store;
    const auto now = Clock::now();

    store.set_history_capacity(core::ProductId::Depth, 4);
    store.publish(make_metric_product(50, now - std::chrono::milliseconds{30}));
    store.publish(make_metric_product(52, now + std::chrono::milliseconds{20}));

    const auto semantic = make_semantic_metadata(51, now);

    perception::Object3DAssociationPolicy policy{};
    policy.max_source_delta = std::chrono::milliseconds{50};

    const auto match = perception::find_metric_observation<int>(store,
                                                                core::ProductId::Depth,
                                                                semantic,
                                                                policy);

    ASSERT_TRUE(match.matched());
    ASSERT_NE(match.product, nullptr);

    EXPECT_EQ(match.method, perception::Object3DMatchMethod::NearestTimestamp);
    EXPECT_EQ(match.product->metadata.observation.sequence, 52U);
    EXPECT_EQ(match.source_delta, std::chrono::milliseconds{20});
}

TEST(Object3DAssociationTest, RejectsWrongSource) {
    core::ProductStore store;
    const auto now = Clock::now();

    store.set_history_capacity(core::ProductId::Depth, 2);
    store.publish(make_metric_product(60, now, core::SourceId::Rplidar));
    const auto semantic = make_semantic_metadata(60, now, core::SourceId::StereoCamera);

    const auto match = perception::find_metric_observation<int>(store,
                                                                core::ProductId::Depth,
                                                                semantic,
                                                                {});

    EXPECT_FALSE(match.matched());
    EXPECT_EQ(match.rejection, perception::Object3DRejectReason::WrongSource);
}

TEST(Object3DAssociationTest, RejectsObservationOutsideTimeBound) {
    core::ProductStore store;
    const auto now = Clock::now();

    store.set_history_capacity(core::ProductId::Depth, 2);
    store.publish(make_metric_product(80, now - std::chrono::milliseconds{100}));

    const auto semantic = make_semantic_metadata(81, now);

    perception::Object3DAssociationPolicy policy{};
    policy.max_source_delta = std::chrono::milliseconds{50};

    const auto match = perception::find_metric_observation<int>(store,
                                                                core::ProductId::Depth,
                                                                semantic,
                                                                policy);

    EXPECT_FALSE(match.matched());
    EXPECT_EQ(match.rejection, perception::Object3DRejectReason::OutsideTimeBound);
    EXPECT_EQ(match.source_delta, std::chrono::milliseconds{100});
}

TEST(Object3DAssociationTest, RejectsWhenNoMetricObservationExists) {
    core::ProductStore store;
    const auto now = Clock::now();
    const auto semantic = make_semantic_metadata(90, now);

    const auto match = perception::find_metric_observation<int>(store,
                                                                core::ProductId::Depth,
                                                                semantic,
                                                                {});

    EXPECT_FALSE(match.matched());
    EXPECT_EQ(match.rejection, perception::Object3DRejectReason::InvalidMetricObservation);
}