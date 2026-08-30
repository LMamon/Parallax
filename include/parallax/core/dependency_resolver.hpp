#pragma once

#include <parallax/core/graph.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace parallax::core {

    enum class DemandSource : std::uint8_t {
        RuntimeBaseline, Application, FoxgloveSubscriber, InternalDependent
    };

    struct DemandSnapshot {
        std::uint64_t revision = 0;
        std::vector<ProductId> products;
    };

    // Resolves requested products into the minimum producer subgraph required.
    //
    // Demand accounting is separate from execution. Acquiring demand records
    // intent only; Runtime decides when producers are considered.
    class DependencyResolver {
        public:
            using ProducerList = std::vector<Producer*>;

            explicit DependencyResolver(const Graph& graph) noexcept : graph_(graph) {}

            [[nodiscard]] ProducerList resolve(ProductId product) const;

            // Resolve the union of dependencies required by several products.
            // Shared producers appear only once.
            [[nodiscard]] ProducerList resolve(const std::vector<ProductId>& products) const;

            void acquire(ProductId product, DemandSource source);
            void release(ProductId product, DemandSource source);

            [[nodiscard]] std::size_t demand(ProductId product, DemandSource source) const noexcept;
            [[nodiscard]] std::size_t total_demand(ProductId product) const noexcept;
            [[nodiscard]] bool demanded(ProductId product) const noexcept {
                return total_demand(product) != 0;
            }

            /**
             * Snapshot active root demand for Runtime.
             *
             * revision changes only when a product enters or leaves the active
             * root set. Reference-count changes that leave the root active do
             * not require rebuilding the execution plan.
             */
            [[nodiscard]] DemandSnapshot active_demand() const;

        private:
            struct ProductIdHash {
                std::size_t operator()(ProductId id) const noexcept {
                    return static_cast<std::size_t>(id);
                }
            };

            struct DemandCounts {
                std::size_t runtime_baseline = 0;
                std::size_t application = 0;
                std::size_t foxglove_subscriber = 0;
                std::size_t internal_dependent = 0;
            };

            [[nodiscard]] static std::size_t& count_for(DemandCounts& counts, DemandSource source) noexcept;
            [[nodiscard]] static const std::size_t& count_for(const DemandCounts& counts, DemandSource source) noexcept;
            [[nodiscard]] static std::size_t total(const DemandCounts& counts) noexcept;

            const Graph& graph_;

            // Demand callbacks can arrive from Foxglove while Runtime reads
            // active roots. Keep only the accounting snapshot under this lock.
            mutable std::mutex demand_mutex_;
            std::unordered_map<ProductId, DemandCounts, ProductIdHash> demand_;

            std::uint64_t active_revision_ = 0;
    };
}