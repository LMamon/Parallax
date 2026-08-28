#pragma once

#include <parallax/core/product.hpp>

#include <chrono>
#include <cstdint>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace parallax::core {
    /**
     * Freshness answers only one question:
     *
     *     Is this product recent enough to still be useful?
     *
     * Source identity and frame association are compatibility concerns and
     * intentionally do not belong here.
     */
    struct FreshnessConstraint {
        std::optional<std::chrono::steady_clock::duration> max_age{};
    };

    /**
     * Test whether a product was produced from one exact source observation.
     *
     * This is a strong compatibility rule appropriate when two products must
     * describe the same sensor observation. Consumers needing looser temporal
     * association should implement that policy themselves rather than teaching
     * ProductStore domain-specific semantics.
     */
    template <typename T> [[nodiscard]] bool same_source_observation(const Product<T>& product,
                                                                     const SourceObservation& required) noexcept {

        return product.metadata.observation == required;
    }

    /**
     * latest-value storage for completed graph products
     * this is not a frame queue. publishing another product with the same ProductId
     * replaces what the store considers current. consumers that already hold an 
     * older Product<T> keep its shared payload alive
     */
    class ProductStore {
        public:
            ProductStore() = default;
            ProductStore(const ProductStore&) = delete;
            ProductStore& operator=(const ProductStore&) = delete;

            template <typename T> void publish(Product<T> product) {
                /**
                 * ProductStore is shared by independently scheduled producers.
                 *
                 * Publication mutates both the latest-value map and optional history,
                 * so those changes occur under one exclusive lock. Consumers either
                 * observe the previous publication or the new publication; they never
                 * observe a partially updated store state.
                 */
                std::unique_lock lock(mutex_);

                auto stored = std::make_shared<Product<T>>(std::move(product));
                const ProductId id = stored->id;

                Entry entry{std::type_index(typeid(T)), stored, stored->metadata};

                entries_[id] = entry;

                const auto history_it = histories_.find(id);
                if (history_it == histories_.end() || history_it->second.capacity == 0) {
                    return;
                }

                auto& history = history_it->second;
                history.entries.push_back(std::move(entry));

                while (history.entries.size() > history.capacity) {
                    history.entries.pop_front();
                }
            }
            
            template <typename T> [[nodiscard]] std::shared_ptr<const Product<T>> latest(ProductId id) const {
                std::shared_lock lock(mutex_);
                return latest_unlocked<T>(id);
            }

            template <typename T> [[nodiscard]] std::shared_ptr<const Product<T>> latest_fresh(ProductId id,
                                                                                               const FreshnessConstraint& freshness,
                                                                                               std::chrono::steady_clock::time_point now) const {

                std::shared_lock lock(mutex_);

                const auto product = latest_unlocked<T>(id);

                if (!product || !product->valid()) return {};
                if (!freshness.max_age.has_value()) return product;
                if (product->metadata.timestamp > now) return {};

                const auto age = now - product->metadata.timestamp;

                if (age > *freshness.max_age) return {};

                return product;
            }

            // a capacity of 0 disables history for the ProductId.
            void set_history_capacity(ProductId, std::size_t capacity);
            
            [[nodiscard]] std::size_t history_capacity(ProductId id) const noexcept;

            template <typename T>[[nodiscard]] std::vector<std::shared_ptr<const Product<T>>> history(ProductId id) const {
                std::shared_lock lock(mutex_);

                const auto it = histories_.find(id);
                if (it == histories_.end()) return {};

                std::vector<std::shared_ptr<const Product<T>>> result;
                result.reserve(it->second.entries.size());

                for (const auto& entry : it->second.entries) {
                    if (entry.type != std::type_index(typeid(T))) {
                        return {};
                    }

                    result.push_back(std::static_pointer_cast<const Product<T>>(entry.product));
                }
                return result;
            }

            template <typename T> [[nodiscard]] std::shared_ptr<const Product<T>> next_after(ProductId id,
                                                                                             const SourceObservation& observation) const {

                std::shared_lock lock(mutex_);

                const auto it = histories_.find(id);
                if (it == histories_.end()) return {};

                for (const auto& entry : it->second.entries) {
                    if (entry.type != std::type_index(typeid(T))) {
                        return {};
                    }

                    const auto product = std::static_pointer_cast<const Product<T>>(entry.product);

                    if (!product || product->metadata.observation.source != observation.source) {
                        continue;
                    }

                    if (product->metadata.observation.sequence > observation.sequence) {
                        return product;
                    }
                }
                return {};
            }            

            [[nodiscard]] bool contains(ProductId id) const noexcept;
            [[nodiscard]] std::optional<ProductMetadata> metadata(ProductId id) const noexcept;                                                                             
            void clear() noexcept;

        private:
            struct ProductIdHash {
                [[nodiscard]] std::size_t operator()(ProductId id) const noexcept {
                    return static_cast<std::size_t>(id);
                }
            };
            
            struct Entry {
                std::type_index type{typeid(void)};
                std::shared_ptr<const void> product{};
                ProductMetadata metadata{};
            };

            struct History {
                std::size_t capacity = 0;
                std::deque<Entry> entries{};
            };

            template <typename T> [[nodiscard]] std::shared_ptr<const Product<T>> latest_unlocked(ProductId id) const {
                const auto it = entries_.find(id);

                if (it == entries_.end()) return {};
                if (it->second.type != std::type_index(typeid(T))) return {};

                return std::static_pointer_cast<const Product<T>>(it->second.product);
            }

            // one entry per ProductId on purpose. historu in opt-in later rather than 
            // making every realtime product accumulate frames by default
            
            // replacing the current product only releases the stores reference.
            // it does not invalidate an accelerator allocation that a consumer is
            // still using because Product<T> keeps the payload behind shared ownership.
            mutable std::shared_mutex mutex_;
            std::unordered_map<ProductId, Entry, ProductIdHash> entries_;
            std::unordered_map<ProductId, History, ProductIdHash> histories_;
        };
}