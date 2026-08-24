#pragma once

#include <parallax/core/graph.hpp>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace parallax::core {
    
    enum class DemandSource : std::uint8_t {
        Application, FoxgloveSubscriber, InternalDependent
    };

    // resolves requested products into the minimum producer subgraph required

    // Demand accounting is separate from execution. Acquiring demand does not
    // submit producers; it only records that a product is required.
    class DependencyResolver {
        public:
            using ProducerList = std::vector<Producer*>;

            explicit DependencyResolver(const Graph& graph) noexcept : graph_(graph) {}
            [[nodiscard]] ProducerList resolve(ProductId product) const;

            // Resolves union of dependencies required by several producs.
            // shared producers appear only once
            [[nodiscard]] ProducerList resolve(const std::vector<ProductId>& products) const;

            // +1 reference from a demand source
            void acquire(ProductId product, DemandSource source);
            // -1 reference from a demand source
            void release(ProductId product, DemandSource source);

            [[nodiscard]] std::size_t demand(ProductId product, DemandSource source) const noexcept;
            [[nodiscard]] std::size_t total_demand(ProductId product) const noexcept;
            [[nodiscard]] bool demanded(ProductId product) const noexcept {
                return total_demand(product) != 0;
            }

        private:
            struct ProductIdHash {
                std::size_t operator()(ProductId id) const noexcept {
                    return static_cast<std::size_t>(id);
                }
            };

            struct DemandCounts {
                std::size_t application = 0;
                std::size_t foxglove_subscriber = 0;
                std::size_t internal_dependent = 0;
            };

            [[nodiscard]] static std::size_t& count_for(DemandCounts& counts, DemandSource source) noexcept;
            [[nodiscard]] static const std::size_t& count_for(const DemandCounts& counts, DemandSource source) noexcept;

            const Graph& graph_;

            std::unordered_map<ProductId, DemandCounts, ProductIdHash> demand_;
    };
}