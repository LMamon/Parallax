#include <parallax/core/fixed_payload_pool.hpp>
#include <parallax/core/product_store.hpp>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <cstdint>
#include <memory>

namespace parallax::core {
    namespace {

        using Clock = std::chrono::steady_clock;
        using TestPayload = int;

        using parallax::core::ProductId;
        using parallax::core::ProductMetadata;
        using parallax::core::ProductStore;
        using parallax::core::SourceId;
        using parallax::core::SourceObservation;

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

        TEST(ProductStoreTest, ConcurrentPublishAndReadIsSafe) {
            ProductStore store;

            constexpr std::uint64_t iterations = 10000;

            std::atomic<bool> writer_done{false};
            std::atomic<bool> reader_failed{false};

            std::thread writer([&]() {
                for (std::uint64_t sequence = 1; sequence <= iterations; ++sequence) {
                    store.publish(make_test_product(ProductId::Depth, sequence, Clock::now(), static_cast<int>(sequence)));
                }

                writer_done.store(true, std::memory_order_release);
            });

            std::thread reader([&]() {
                std::uint64_t last_sequence = 0;

                while (!writer_done.load(
                    std::memory_order_acquire)) {

                    const auto product = store.latest<TestPayload>(ProductId::Depth);

                    if (!product) continue;

                    const auto sequence = product->metadata.observation.sequence;
                    if (sequence < last_sequence) {
                        reader_failed.store(true);
                        return;
                    }

                    if (!product->payload || *product->payload != static_cast<int>(sequence)) {
                        reader_failed.store(true);
                        return;
                    }

                    last_sequence = sequence;
                }
            });

            writer.join();
            reader.join();

            EXPECT_FALSE(reader_failed.load());

            const auto latest = store.latest<TestPayload>(ProductId::Depth);

            ASSERT_NE(latest, nullptr);
            EXPECT_EQ(latest->metadata.observation.sequence, iterations);
        }

        TEST(FixedPayloadPoolTest, CapacityIsBounded) {
            FixedPayloadPool<int, 2> pool;

            ASSERT_TRUE(pool.initialize([](int& slot, std::size_t index) {
                slot = static_cast<int>(index);
                return true;
            }));

            EXPECT_EQ(pool.capacity(), 2);

            auto first = pool.acquire();
            auto second = pool.acquire();

            ASSERT_NE(first, nullptr);
            ASSERT_NE(second, nullptr);
            EXPECT_NE(first.get(), second.get());

            // Both preallocated slots are leased. acquire() must not allocate
            // another payload or grow the pool.
            auto exhausted = pool.acquire();

            EXPECT_EQ(exhausted, nullptr);
        }

        TEST(FixedPayloadPoolTest, HeldPayloadCannotBeReused) {
            FixedPayloadPool<int, 2> pool;

            ASSERT_TRUE(pool.initialize([](int& slot, std::size_t index) {
                slot = static_cast<int>(index);
                return true;
            }));

            auto held = pool.acquire();
            ASSERT_NE(held, nullptr);

            int* const held_address = held.get();
            auto other = pool.acquire();

            ASSERT_NE(other, nullptr);
            EXPECT_NE(other.get(), held_address);

            // Both slots are currently leased.
            EXPECT_EQ(pool.acquire(), nullptr);

            other.reset();
            auto reacquired = pool.acquire();

            ASSERT_NE(reacquired, nullptr);
            // The held slot is still protected by shared ownership, so the only
            // reusable slot must be the one just released.
            EXPECT_NE(reacquired.get(), held_address);
        }

        TEST(FixedPayloadPoolTest, ReleasedPayloadBecomesReusable) {
            FixedPayloadPool<int, 1> pool;

            ASSERT_TRUE(pool.initialize([](int& slot, std::size_t) {
                slot = 42;
                return true;
            }));

            auto first = pool.acquire();

            ASSERT_NE(first, nullptr);
            int* const address = first.get();

            EXPECT_EQ(pool.acquire(), nullptr);

            first.reset();
            auto second = pool.acquire();

            ASSERT_NE(second, nullptr);
            // No allocation/growth occurred. The original preallocated slot was
            // returned to the reusable set once its lease disappeared.
            EXPECT_EQ(second.get(), address);
            EXPECT_EQ(*second, 42);
        }

        TEST(ProductStoreTest, NextAfterReturnsOldestNewerRetainedObservation) {
            ProductStore store;
            const auto now = Clock::now();

            store.set_history_capacity(ProductId::Depth, 4);

            store.publish(make_test_product(ProductId::Depth, 10, now, 10));
            store.publish(make_test_product(ProductId::Depth, 11, now, 11));
            store.publish(make_test_product(ProductId::Depth, 12, now, 12));

            const SourceObservation consumed{SourceId::StereoCamera, 10};

            const auto next = store.next_after<TestPayload>(ProductId::Depth, consumed);

            ASSERT_NE(next, nullptr);
            EXPECT_EQ(next->metadata.observation.sequence, 11);
            EXPECT_EQ(*next->payload, 11);
        }


        TEST(ProductStoreTest, NextAfterDoesNotCrossSourceDomains) {
            ProductStore store;
            const auto now = Clock::now();

            store.set_history_capacity(ProductId::Depth, 4);

            store.publish(make_test_product(ProductId::Depth, 10, now, 10, SourceId::StereoCamera));
            store.publish(make_test_product(ProductId::Depth, 11, now, 11, SourceId::Rplidar));

            const SourceObservation consumed{SourceId::StereoCamera, 10};

            const auto next = store.next_after<TestPayload>(ProductId::Depth, consumed);
            EXPECT_EQ(next, nullptr);
        }

        TEST(ProductStoreTest, NextAfterReturnsNullWhenNoNewerObservationIsRetained) {
            ProductStore store;
            const auto now = Clock::now();

            store.set_history_capacity(ProductId::Depth, 2);

            store.publish(make_test_product(ProductId::Depth, 10, now, 10));
            store.publish(make_test_product(ProductId::Depth, 11, now, 11));

            const SourceObservation consumed{SourceId::StereoCamera, 11};

            const auto next = store.next_after<TestPayload>(ProductId::Depth, consumed);

            EXPECT_EQ(next, nullptr);
        }

        TEST(ProductStoreTest, FindObservationReturnsExactRetainedGeneration) {
            ProductStore store;
            store.set_history_capacity(ProductId::RgbLeft, 3);

            for (std::uint64_t sequence = 40; sequence <= 42; ++sequence) {
                ProductMetadata metadata{};
                metadata.observation = SourceObservation{SourceId::StereoCamera, sequence};
                metadata.valid = true;

                std::shared_ptr<const int> payload = std::make_shared<int>(static_cast<int>(sequence));

                store.publish(parallax::core::make_product<int>(ProductId::RgbLeft,
                                                           metadata,
                                                           std::move(payload)));
            }

            const auto found = store.find_observation<int>(ProductId::RgbLeft, SourceObservation{SourceId::StereoCamera, 41});

            ASSERT_NE(found, nullptr);
            EXPECT_EQ(found->metadata.observation.source, SourceId::StereoCamera);
            EXPECT_EQ(found->metadata.observation.sequence, 41U);
            ASSERT_NE(found->payload, nullptr);
            EXPECT_EQ(*found->payload, 41);
        }

        TEST(ProductStoreTest, FindObservationCanReturnCurrentGenerationWithoutHistory) {
            ProductStore store;

            ProductMetadata metadata{};
            metadata.observation = SourceObservation{SourceId::StereoCamera, 52};
            metadata.valid = true;

            store.publish(parallax::core::make_product<int>(ProductId::RgbLeft,
                                                       metadata,
                                                       std::make_shared<int>(52)));

            const auto found = store.find_observation<int>(ProductId::RgbLeft,
                                                           SourceObservation{SourceId::StereoCamera, 52});

            ASSERT_NE(found, nullptr);
            EXPECT_EQ(found->metadata.observation.sequence, 52U);
            EXPECT_EQ(*found->payload, 52);
        }

        TEST(ProductStoreTest, FindObservationRejectsDifferentSource) {
            ProductStore store;
            store.set_history_capacity(ProductId::RgbLeft, 2);

            ProductMetadata metadata{};
            metadata.observation = SourceObservation{SourceId::StereoCamera, 60};
            metadata.valid = true;

            store.publish(parallax::core::make_product<int>(ProductId::RgbLeft,
                                                       metadata,
                                                       std::make_shared<int>(60)));

            const auto found = store.find_observation<int>(ProductId::RgbLeft, SourceObservation{SourceId::Rplidar, 60});

            EXPECT_EQ(found, nullptr);
        }

        TEST(ProductStoreTest, FindObservationFailsAfterGenerationLeavesBoundedHistory) {
            ProductStore store;
            store.set_history_capacity(ProductId::RgbLeft, 2);

            for (std::uint64_t sequence = 70; sequence <= 72; ++sequence) {
                ProductMetadata metadata{};
                metadata.observation = SourceObservation{SourceId::StereoCamera, sequence};
                metadata.valid = true;

                store.publish(parallax::core::make_product<int>(ProductId::RgbLeft,
                                                           metadata,
                                                           std::make_shared<int>(static_cast<int>(sequence))));
            }

            const auto found = store.find_observation<int>(ProductId::RgbLeft, SourceObservation{SourceId::StereoCamera, 70});
            EXPECT_EQ(found, nullptr);
        }
    }
} 