#pragma once

#include <parallax/core/graph.hpp>
#include <parallax/core/product_store.hpp>

namespace parallax::core {

    inline void configure_product_history(const Graph& graph, ProductStore& products) {
        for (Producer* producer : graph.producers()) {
            if (producer == nullptr) continue;

            for (const auto& requirement : producer->ordered_inputs()) {
                const std::size_t existing = products.history_capacity(requirement.product);

                if (requirement.history_capacity > existing) {
                    products.set_history_capacity(requirement.product, requirement.history_capacity);
                }
            }

            for (const auto& requirement : producer->compatible_inputs()) {
                const std::size_t existing = products.history_capacity(requirement.product);

                if (requirement.history_capacity > existing) {
                    products.set_history_capacity(requirement.product, requirement.history_capacity);
                }
            }
        }
    }
}