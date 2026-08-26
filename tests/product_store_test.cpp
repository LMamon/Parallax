#include <parallax/core/product_store.hpp>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>

namespace parallax::core {
    namespace {
        using Clock = std::chrono::steady_clock;
        // simple CPU payload for testing store semantics. the store should not care
        // whether T is an int like this or an accelerator-backed resource.
        using TestPayload = int;
        Product<TestPayload> make_test_product(ProductId id, 
                                               std::uint64_t sequence, 
                                               Clock::time_point timestamp,
                                               int value) {
            
            ProductMetadata metadata{};
            metadata.observation.sequence = sequence;
            metadata.timestamp = timestamp;
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
            ASSERT_NE(latest->payload, nullptr);
            EXPECT_EQ(*latest->payload, 20);
        }

        TEST(ProductStoreTest, LatestCompatibleRejectsWrongSequence) {
            ProductStore store;
            const auto now = Clock::now();

            store.publish(make_test_product(ProductId::Depth, 10, now, 42));

            FreshnessConstraint freshness{};
            freshness.observation = SourceObservation{SourceId::StereoCamera, 11};

            const auto result = store.latest_compatible<TestPayload>(ProductId::Depth, freshness, now);

            EXPECT_EQ(result, nullptr);
        }

        TEST(ProductStoreTest, LatestCompatibleRejectsStaleProduct) {
            ProductStore store;

            const auto now = Clock::now();
            store.publish(make_test_product(ProductId::Depth, 1, now - std::chrono::milliseconds{100}, 42));

            FreshnessConstraint freshness{};
            freshness.max_age = std::chrono::milliseconds{50};

            const auto result = store.latest_compatible<TestPayload>(ProductId::Depth, freshness, now);

            EXPECT_EQ(result, nullptr);
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
            // publishing sequence 2 replaces the store's latest Depth entry.
            // old_product still owns sequence 1 through shared ownership, which
            // is the same lifetime behavior needed for CUDA/VPI-backed payloads.
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