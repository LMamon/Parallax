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

    enum class SubmissionDecision {
        Submit, StaleInput, RateLimited, Superseded
    };

    [[nodiscard]] inline bool is_compatible_input(const Producer& producer, ProductId product) noexcept {
        for (const auto& requirement : producer.compatible_inputs()) {
            if (requirement.product == product) {
                return true;
            }
        }

        return false;
    }

    /**
     * Returns the single source observation described by all producer inputs.
     *
     * Source producers have no input observation and return nullopt.
     * Missing, invalid, or mutually incompatible inputs also return nullopt.
     */
    [[nodiscard]] inline std::optional<SourceObservation> input_observation(const Producer& producer, const ProductStore& products) {
        const auto& inputs = producer.inputs();
        if (inputs.empty()) return std::nullopt;

        std::optional<SourceObservation> strict_observation;
        std::optional<SourceObservation> fallback_observation;

        for (const ProductId input : inputs) {
            const auto metadata = products.metadata(input);

            if (!metadata || !metadata->valid || !metadata->observation.valid()) {
                return std::nullopt;
            }

            if (!fallback_observation) fallback_observation = metadata->observation;

            /*
            * Compatible inputs are selected by the producer from bounded history.
            * Their latest generation therefore does not have to match the strict
            * scheduling observation.
            */
            if (is_compatible_input(producer, input)) continue;

            if (!strict_observation) {
                strict_observation = metadata->observation;
                continue;
            }

            if (*strict_observation != metadata->observation) return std::nullopt;
        }

        return strict_observation ? strict_observation : fallback_observation;
    }

    [[nodiscard]] inline std::optional<InputObservation> input_observation_with_timestamp(const Producer& producer,
                                                                                          const ProductStore& products) {

        const auto& inputs = producer.inputs();
        if (inputs.empty()) return std::nullopt;

        std::optional<InputObservation> strict_result;
        std::optional<InputObservation> fallback_result;

        for (const ProductId input : inputs) {
            const auto metadata = products.metadata(input);

            if (!metadata || !metadata->valid || !metadata->observation.valid()) {
                return std::nullopt;
            }

            if (!fallback_result) {
                fallback_result = InputObservation{metadata->observation, metadata->timestamp};
            }

            /*
            * Compatible generations are selected inside the producer. The latest
            * compatible input must neither reject the strict observation nor make
            * its scheduling timestamp artificially older.
            */
            if (is_compatible_input(producer, input)) continue;

            if (!strict_result) {
                strict_result = InputObservation{metadata->observation, metadata->timestamp};
                continue;
            }

            if (strict_result->observation != metadata->observation) return std::nullopt;

            if (metadata->timestamp < strict_result->timestamp) {
                strict_result->timestamp = metadata->timestamp;
            }
        }
        return strict_result ? strict_result : fallback_result;
    }


    [[nodiscard]] inline SubmissionDecision submission_decision(const ExecutionPolicy& policy,
                                                                const ProducerExecutionState& state,
                                                                const InputObservation& input,
                                                                std::chrono::steady_clock::time_point now) noexcept {

        if (policy.max_input_age_ms > 0.0) {
            if (input.timestamp > now) {
                return SubmissionDecision::StaleInput;
            }

            const auto age_ms = std::chrono::duration<double, std::milli>(now - input.timestamp).count();
            if (age_ms > policy.max_input_age_ms) {
                return SubmissionDecision::StaleInput;
            }
        }

        if (policy.target_hz > 0.0 && state.has_last_submission) {
            const auto elapsed = std::chrono::duration<double>(now - state.last_submission).count();
            if (elapsed < (1.0 / policy.target_hz)) {
                return SubmissionDecision::RateLimited;
            }
        }

        if (policy.drop_policy == DropPolicy::Supersede && state.has_last_observation && state.last_observation == input.observation) {
            return SubmissionDecision::Superseded;
        }

        return SubmissionDecision::Submit;
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
}