#pragma once

#include <parallax/core/product.hpp>
#include <parallax/core/product_store.hpp>
#include <parallax/perception/image_space.hpp>

#include <chrono>
#include <cstdint>
#include <memory>

namespace parallax::perception {

    enum class Object3DMatchMethod : std::uint8_t {
        None = 0,
        ExactObservation,
        NearestTimestamp
    };

    enum class Object3DRejectReason : std::uint8_t {
        None = 0,
        WrongSource,
        WrongImageSpace,
        OutsideTimeBound,
        InvalidMetricObservation,
        InsufficientSupport
    };

    struct Object3DAssociationPolicy {
        std::chrono::steady_clock::duration max_source_delta = std::chrono::milliseconds{50};
    };

    template <typename T> struct Object3DMatch {
        std::shared_ptr<const core::Product<T>> product{};
        Object3DMatchMethod method = Object3DMatchMethod::None;
        Object3DRejectReason rejection = Object3DRejectReason::None;
        std::chrono::steady_clock::duration source_delta{};

        [[nodiscard]] bool matched() const noexcept {
            return static_cast<bool>(product) && method != Object3DMatchMethod::None && rejection == Object3DRejectReason::None;
        }
    };

    template <typename T> [[nodiscard]] Object3DMatch<T> find_metric_observation(const core::ProductStore& products,
                                                                                 core::ProductId metric_product,
                                                                                 const core::ProductMetadata& semantic_metadata,
                                                                                 ImageSpace semantic_image_space,
                                                                                 ImageSpace required_image_space,
                                                                                 const Object3DAssociationPolicy& policy) {

        Object3DMatch<T> result{};

        if (!semantic_metadata.valid || !semantic_metadata.observation.valid()) {
            result.rejection = Object3DRejectReason::InvalidMetricObservation;
            return result;
        }

        if (semantic_image_space != required_image_space) {
            result.rejection = Object3DRejectReason::WrongImageSpace;
            return result;
        }

        const auto exact = products.find_observation<T>(metric_product, semantic_metadata.observation);

        if (exact && exact->valid()) {
            result.product = exact;
            result.method = Object3DMatchMethod::ExactObservation;
            result.source_delta = exact->metadata.timestamp - semantic_metadata.timestamp;
            return result;
        }

        const auto history = products.history<T>(metric_product);

        std::shared_ptr<const core::Product<T>> nearest{};
        auto nearest_delta = std::chrono::steady_clock::duration::max();
        bool saw_wrong_source = false;
        bool saw_same_source = false;

        for (const auto& candidate : history) {
            if (!candidate || !candidate->valid()) continue;

            if (candidate->metadata.observation.source != semantic_metadata.observation.source) {
                saw_wrong_source = true;
                continue;
            }

            saw_same_source = true;

            const auto delta = candidate->metadata.timestamp >= semantic_metadata.timestamp
                               ? candidate->metadata.timestamp - semantic_metadata.timestamp
                               : semantic_metadata.timestamp - candidate->metadata.timestamp;

            if (delta < nearest_delta) {
                nearest = candidate;
                nearest_delta = delta;
            }
        }

        if (!nearest) {
            result.rejection = saw_wrong_source && !saw_same_source
                                ? Object3DRejectReason::WrongSource
                                : Object3DRejectReason::InvalidMetricObservation;
            return result;
        }

        if (nearest_delta > policy.max_source_delta) {
            result.rejection = Object3DRejectReason::OutsideTimeBound;
            result.source_delta = nearest_delta;
            return result;
        }

        result.product = nearest;
        result.method = Object3DMatchMethod::NearestTimestamp;
        result.source_delta = nearest_delta;
        return result;
    }
}