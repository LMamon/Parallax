#pragma once

#include <parallax/core/product_id.hpp>
#include <parallax/core/completion.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>

namespace parallax::core {

    /**
     * Identifies the observation/clock domain in which a source sequence
     * has meaning.
     *
     * Sequence values are only comparable when their SourceId matches.
     * Camera sequence 42 and LiDAR sequence 42 therefore describe unrelated
     * observations.
     */
    enum class SourceId : std::uint8_t {
        Unknown = 0, StereoCamera, Rplidar
    };

    /**
     * Identity of the source observation from which a product was derived.
     *
     * Derived products preserve this identity unchanged until a producer
     * intentionally combines multiple independent source observations.
     */
    struct SourceObservation {
        SourceId source = SourceId::Unknown;
        std::uint64_t sequence = 0;

        [[nodiscard]] bool valid() const noexcept {
            return source != SourceId::Unknown;
        }

        [[nodiscard]] bool operator==(const SourceObservation& other) const noexcept {
            return source == other.source && sequence == other.sequence;
        }

        [[nodiscard]] bool operator!=(const SourceObservation& other) const noexcept {
            return !(*this == other);
        }
    };

    /**
     * Metadata that follows a product through the graph.
     *
     * observation identifies where the data originated.
     * timestamp is the capture/observation time in the local steady-clock domain.
     * production_timestamp records when this particular product representation
     * was published and is useful for later end-to-end latency instrumentation.
     */
    struct ProductMetadata {
        SourceObservation observation{};

        std::chrono::steady_clock::time_point timestamp{};
        std::chrono::steady_clock::time_point production_timestamp{};

        bool valid = false;
    };

    template <typename T> struct Product {
        ProductId id{};
        ProductMetadata metadata{};
        CompletionHandle completion{};
        std::shared_ptr<const T> payload{};

        [[nodiscard]] bool valid() const noexcept {
            return metadata.valid && static_cast<bool>(payload);
        }

        [[nodiscard]] explicit operator bool() const noexcept {
            return valid();
        }
    };

    template <typename T> [[nodiscard]] Product<T> make_product(ProductId id,
                                                                ProductMetadata metadata,
                                                                std::shared_ptr<const T> payload,
                                                                CompletionHandle completion = CompletionHandle::cpu_ready()) {

        return Product<T>{id, metadata, std::move(completion), std::move(payload)};
    }
}