#pragma once

#include <parallax/core/producer.hpp>

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace parallax::core {
    /**
     * Graph does not own producers and does not execute them. It records the
     * dependency structure declared by Producer::inputs() / outputs() so later
     * resolution can determine the minimum producer subgraph for a request.
     *
     * Producer lifetimes must outlive the Graph.
     */
    class Graph {

        public:
            using ProducerList = std::vector<Producer*>;

            // Registers a producer and maps each of its outputs to that producer.
            // Throws std::logic_error if an output already has a producer.
            void register_producer(Producer& producer);
        
            // Finalizes producer dependency edges and verifies that the graph is
            // acyclic. Call after all producers have been registered.
            // Throws std::logic_error when a dependency cycle is detected.
            void finalize();
            [[nodiscard]] Producer* producer_for(ProductId product) const noexcept;

            // Direct upstream producers required by this producer.
            [[nodiscard]] const ProducerList& dependencies_of(const Producer& producer) const noexcept;

            [[nodiscard]] const ProducerList& producers() const noexcept {
                return producers_;
            }

        private:

            struct ProductIdHash {
                std::size_t operator()(ProductId id) const noexcept {
                    return static_cast<std::size_t>(id);
                }
            };

            void verify_acyclic() const;

            ProducerList producers_;

            std::unordered_map<ProductId, Producer*, ProductIdHash>product_producers_;
            std::unordered_map<const Producer*, ProducerList>dependencies_;
    };
}