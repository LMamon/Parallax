#include <parallax/core/product_store.hpp>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>

namespace parallax::core {
    namespace {

        using Clock = std::chrono::steady_clock;
        using TestPayload = int;

        /**
         * Simple CPU payload for testing ProductStore semantics.
         *
         * Source identity is explicit because sequence numbers only have
         * meaning within a source domain.
         */
        Product<TestPayload> make_test_product(ProductId id,
                                               std::uint64_t sequence,
                                               Clock::time_point timestamp,
                                               int value,
                                               SourceId source = SourceId::StereoCamera) {

            ProductMetadata metadata{};
            metadata.observation.source = source;
            metadata.observation.sequence = sequence;
            metadata.timestamp = timestamp;
            metadata.production_timestamp = Clock::now();
            metadata.valid = true;

            return make_product<TestPayload>(id, metadata, std::make_shared<const TestPayload>(value));
        }

        TEST(ProductStoreTest, LatestReturnsMostRecentlyPublishedProduct) {
            ProductStore store;
            const auto now = Clock::now();

            store.publish(make_test_product(ProductId::Depth, 1, now, 10));
            store.publish(make_test_product(ProductId::Depth, 2, now, 20));

            const auto latest = store.latest<TestPayload>(ProductId::Depth);

            ASSERT_NE(latest, nullptr);
            EXPECT_EQ(latest->metadata.observation.sequence, 2);
            EXPECT_EQ(latest->metadata.observation.source, SourceId::StereoCamera);

            ASSERT_NE(latest->payload, nullptr);
            EXPECT_EQ(*latest->payload, 20);
        }

        TEST(ProductStoreTest, LatestFreshAcceptsRecentProduct) {
            ProductStore store;
            const auto now = Clock::now();

            store.publish(make_test_product(ProductId::Depth, 10, now - std::chrono::milliseconds{25}, 42));

            FreshnessConstraint freshness{};
            freshness.max_age = std::chrono::milliseconds{100};

            const auto product = store.latest_fresh<TestPayload>(ProductId::Depth, freshness, now);

            ASSERT_NE(product, nullptr);
            EXPECT_EQ(*product->payload, 42);
        }

        TEST(ProductStoreTest, LatestFreshRejectsStaleProduct) {
            ProductStore store;
            const auto now = Clock::now();

            store.publish(make_test_product(ProductId::Depth, 1, now - std::chrono::milliseconds{100}, 42));
            FreshnessConstraint freshness{};
            freshness.max_age = std::chrono::milliseconds{50};

            const auto product = store.latest_fresh<TestPayload>(ProductId::Depth, freshness, now);

            EXPECT_EQ(product, nullptr);
        }

        TEST(ProductStoreTest, SourceCompatibilityRequiresSameSourceAndSequence) {
            ProductStore store;
            const auto now = Clock::now();

            store.publish(make_test_product(ProductId::Depth, 11, now, 42, SourceId::StereoCamera));

            const auto product = store.latest<TestPayload>(ProductId::Depth);

            ASSERT_NE(product, nullptr);

            const SourceObservation exact{SourceId::StereoCamera, 11};
            const SourceObservation wrong_source{SourceId::Rplidar, 11};
            const SourceObservation wrong_sequence{SourceId::StereoCamera, 12};

            EXPECT_TRUE(same_source_observation(*product, exact));
            EXPECT_FALSE(same_source_observation(*product, wrong_source));
            EXPECT_FALSE(same_source_observation(*product, wrong_sequence));
        }

        TEST(ProductStoreTest, FreshnessDoesNotImplySourceCompatibility) {
            ProductStore store;
            const auto now = Clock::now();

            store.publish(make_test_product(ProductId::Depth, 11, now, 42, SourceId::Rplidar));

            FreshnessConstraint freshness{};
            freshness.max_age = std::chrono::milliseconds{100};

            const auto product = store.latest_fresh<TestPayload>(ProductId::Depth, freshness, now);

            ASSERT_NE(product, nullptr);
            const SourceObservation required{SourceId::StereoCamera, 11};
            
            EXPECT_FALSE(same_source_observation(*product, required));
        }

        TEST(ProductStoreTest, HistoryEvictsOldestProduct) {
            ProductStore store;
            const auto now = Clock::now();

            store.set_history_capacity(ProductId::Depth, 2);

            store.publish(make_test_product(ProductId::Depth, 1, now, 10));
            store.publish(make_test_product(ProductId::Depth, 2, now, 20));
            store.publish(make_test_product(ProductId::Depth, 3, now, 30));

            const auto history = store.history<TestPayload>(ProductId::Depth);

            ASSERT_EQ(history.size(), 2);

            EXPECT_EQ(history[0]->metadata.observation.sequence, 2);
            EXPECT_EQ(history[1]->metadata.observation.sequence, 3);

            EXPECT_EQ(*history[0]->payload, 20);
            EXPECT_EQ(*history[1]->payload, 30);
        }

        TEST(ProductStoreTest, ReplacingLatestDoesNotInvalidateHeldPayload) {
            ProductStore store;
            const auto now = Clock::now();

            store.publish(make_test_product(ProductId::Depth, 1, now, 10));

            const auto old_product = store.latest<TestPayload>(ProductId::Depth);

            ASSERT_NE(old_product, nullptr);
            ASSERT_NE(old_product->payload, nullptr);

            /**
             * Publishing sequence 2 replaces the store's latest Depth entry.
             * old_product still owns sequence 1 through shared ownership,
             * matching the lifetime behavior required for accelerator-backed
             * payloads.
             */
            store.publish(make_test_product(ProductId::Depth, 2, now, 20));

            const auto new_product = store.latest<TestPayload>(ProductId::Depth);

            ASSERT_NE(new_product, nullptr);
            EXPECT_EQ(new_product->metadata.observation.sequence, 2);
            EXPECT_EQ(*new_product->payload, 20);
            EXPECT_EQ(old_product->metadata.observation.sequence, 1);
            EXPECT_EQ(*old_product->payload, 10);
        }
    }
} 