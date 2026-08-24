#pragma once

#include <parallax/core/graph.hpp>

#include <vector>

namespace parallax::core {
    
    // resolves requested products into the minimum producer subgraph required
    class DependencyResolver {
        public:
            using ProducerList = std::vector<Producer*>;

            explicit DependencyResolver(const Graph& graph) noexcept : graph_(graph) {}
            [[nodiscard]] ProducerList resolve(ProductId product) const;

            // Resolves union of dependencies required by several producs.
            // shared producers appear only once
            [[nodiscard]] ProducerList resolve(const std::vector<ProductId>& products) const;

        private:
            const Graph& graph_;
    };
}