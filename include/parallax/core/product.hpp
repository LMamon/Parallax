#pragma once

#include <parallax/core/product_id.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>

namespace parallax::core {
 
    //metadata that follows a product through the graph.
    //sequence+ timespamp explains what source frame a product came from.
    struct ProductMetadata {
        std::uint64_t sequence = 0;
        std::chrono::steady_clock::time_point timestamp{};
        bool valid = false;
    };

    // Product is just the typed payload + metadata needed by the graph
    
    // payload should be kept behind a shared handle. A lot of these products own 
    // CUDA/VPI memory so trying to avoid passing a product around the graph to turn 
    // into another buffer copy.
    template <typename T>
    struct Product {
        ProductId id{};
        ProductMetadata metadata{};
        std::shared_ptr<const T> payload{};

        [[nodiscard]] bool valid() const noexcept {
            return metadata.valid && static_cast<bool>(payload);
        }

        [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
    };

    template <typename T>
    [[nodiscard]] Product<T> make_product(ProductId id, ProductMetadata metadata, std::shared_ptr<const T> payload) {
        return Product<T>{id, metadata, std::move(payload)};
    }
}