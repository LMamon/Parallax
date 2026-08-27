#pragma once

#include <parallax/core/producer.hpp>
#include <parallax/core/product_store.hpp>

#include <optional>

namespace parallax::core {

    struct ProducerExecutionState {
        SourceObservation last_observation{};
        bool has_last_observation = false;

        std::chrono::steady_clock::time_point last_submission{};
        bool has_last_submission = false;
    };

    struct InputObservation {
        SourceObservation observation{};
        std::chrono::steady_clock::time_point timestamp{};
    };

    /**
     * Returns the single source observation described by all producer inputs.
     *
     * Source producers have no input observation and return nullopt.
     * Missing, invalid, or mutually incompatible inputs also return nullopt.
     */
    [[nodiscard]] inline std::optional<SourceObservation> input_observation(const Producer& producer,
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


    [[nodiscard]] inline bool should_submit(const ExecutionPolicy& policy,
                                            const ProducerExecutionState& state,
                                            const InputObservation& input,
                                            std::chrono::steady_clock::time_point now) noexcept {

        if (policy.max_input_age_ms > 0.0) {
            if (input.timestamp > now) {
                return false;
            }

            const auto age_ms = std::chrono::duration<double, std::milli>(now - input.timestamp).count();

            if (age_ms > policy.max_input_age_ms) {
                return false;
            }
        }

        if (policy.target_hz > 0.0 && state.has_last_submission) {
            const auto elapsed = std::chrono::duration<double>(now - state.last_submission).count();

            if (elapsed < (1.0 / policy.target_hz)) {
                return false;
            }
        }

        if (policy.drop_policy == DropPolicy::Supersede &&
            state.has_last_observation &&
            state.last_observation == input.observation) {

            return false;
        }
        return true;
    }


    inline void record_submission(ProducerExecutionState& state,
                                  const ExecutionPolicy& policy,
                                  const InputObservation& input,
                                  std::chrono::steady_clock::time_point now) noexcept {

        state.last_submission = now;
        state.has_last_submission = true;

        if (policy.drop_policy == DropPolicy::Supersede) {
            state.last_observation = input.observation;
            state.has_last_observation = true;
        }
    }

    [[nodiscard]] inline bool input_is_fresh(const ExecutionPolicy& policy,
                                             const ProductMetadata& metadata,
                                             std::chrono::steady_clock::time_point now) noexcept {

        if (policy.max_input_age_ms <= 0.0) {
            return true;
        }

        if (metadata.timestamp > now) {
            return false;
        }

        const auto age_ms = std::chrono::duration<double, std::milli>(now - metadata.timestamp).count();
        return age_ms <= policy.max_input_age_ms;
    }

    [[nodiscard]] inline bool rate_due(const ExecutionPolicy& policy,
                                       const ProducerExecutionState& state,
                                       std::chrono::steady_clock::time_point now) noexcept {

        if (policy.target_hz <= 0.0 || !state.has_last_submission) {
            return true;
        }

        const auto period = std::chrono::duration<double>(1.0 / policy.target_hz);
        return (now - state.last_submission) >= period;
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