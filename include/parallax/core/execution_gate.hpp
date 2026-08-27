#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>

#include <optional>

namespace parallax::core {

    struct ProducerExecutionState {
        SourceObservation last_observation{};
        bool has_last_observation = false;
    };

    /**
     * Returns the single source observation described by all producer inputs.
     *
     * Source producers have no input observation and return nullopt.
     * Missing, invalid, or mutually incompatible inputs also return nullopt.
     */
    [[nodiscard]] inline std::optional<SourceObservation>
    input_observation(
        const Producer& producer,
        const ProductStore& products) {

        const auto& inputs = producer.inputs();

        if (inputs.empty()) {
            return std::nullopt;
        }

        std::optional<SourceObservation> observation;

        for (const ProductId input : inputs) {
            const auto metadata = products.metadata(input);

            if (!metadata ||
                !metadata->valid ||
                !metadata->observation.valid()) {

                return std::nullopt;
            }

            if (!observation) {
                observation = metadata->observation;
                continue;
            }

            if (*observation != metadata->observation) {
                return std::nullopt;
            }
        }

        return observation;
    }

    /**
     * Supersede producers execute only for a source observation they have not
     * already consumed.
     *
     * Block policy is intentionally unaffected here. Ordered/history behavior
     * remains opt-in and is not inferred from latest-value storage.
     */
    [[nodiscard]] inline bool should_submit(
        const ExecutionPolicy& policy,
        const ProducerExecutionState& state,
        const SourceObservation& observation) noexcept {

        if (policy.drop_policy != DropPolicy::Supersede) {
            return true;
        }

        return !state.has_last_observation ||
               state.last_observation != observation;
    }

    inline void record_submission(
        ProducerExecutionState& state,
        const ExecutionPolicy& policy,
        const SourceObservation& observation) noexcept {

        if (policy.drop_policy != DropPolicy::Supersede) {
            return;
        }

        state.last_observation = observation;
        state.has_last_observation = true;
    }

} // namespace parallax::core