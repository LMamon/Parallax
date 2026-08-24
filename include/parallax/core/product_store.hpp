#pragma once

#include <parallax/core/product.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <typeindex>
#include <unordered_map>

namespace parallax::core {
    /**
     * sequence is useful when two products are expected to come from the same
     * source frame. max age is useful when branches run at different rates and
     * exact frame lockstep is unnecessary
     * 
     * leaving either field empty means that constraint is not being requested.
     */
    struct FreshnessConstraint {
        std::optional<std::uint64_t> sequence{};
        std::optional<std::chrono::steady_clock::duration> max_age{};
    };

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

            template <typename T>
            void publish(Product<T> product) {
                // keep the whole Product<T> behind a shared handle so typed erasure
                // here doesnt copy actual CUDA/VPI/etc payload
                auto stored = std::make_shared<Product<T>>(std::move(product));

                entries_[stored->id] = Entry{std::type_index(typeid(T)), std::move(stored)};
            }
            
            template<typename T>
            [[nodiscard]] std::shared_ptr<const Product<T>> latest(ProductId id) const {
                const auto it = entries_.find(id);
                if (it == entries_.end()) return {};

                /**
                 * ProductId describes the semantic product, but multiple product IDs can
                 * eventually use the same/different C++ payload types. check the stored 
                 * type before recovering typed Product<T>
                 */
                if (it->second.type != std::type_index(typeid(T))) return {};

                return std::static_pointer_cast<const Product<T>>(it->second.product);
            }

            template <typename T>
            [[nodiscard]] std::shared_ptr<const Product<T>> latest_compatible(ProductId id,
                                                                              const FreshnessConstraint& freshness,
                                                                              std::chrono::steady_clock::time_point now)
                                                                                    const {
                /**
                 * latest compatible does not search for an older matching product.
                 * the store is still latest-vale onluy at this point. get the current
                 * product first, then either accept/reject it as stale
                 */
                const auto product = latest<T>(id);
                if (!product || !product->valid()) return {};
                
                // when exact sourceframe id matters, accepting a different sequence could associate results from unrelated frames.
                if (freshness.sequence.has_value() && product->metadata.sequence != *freshness.sequence) {
                    return {};
                }

                if (freshness.max_age.has_value()) {
                    // a product timestamp in the future relative to this lookup is not compatible.
                    if (product->metadata.timestamp > now) return {};

                    const auto age = now - product->metadata.timestamp;
                    if (age > *freshness.max_age) return {};
                }
                return product;
            }

            [[nodiscard]] bool contains(ProductId id) const noexcept;
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
            };

            // one entry per ProductId on purpose. historu in opt-in later rather than 
            // making every realtime product accumulate frames by default
            
            // replacing the current product only releases the store's reference.
            // it does not invalidate an accelerator allocation that a consumer is
            // still using because Product<T> keeps the payload behind shared ownership.
            std::unordered_map<ProductId, Entry, ProductIdHash> entries_;
        };
}