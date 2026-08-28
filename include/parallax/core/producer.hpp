#pragma once

#include <parallax/core/execution_policy.hpp>
#include <parallax/core/product.hpp>

#include <string_view>
#include <cstddef>
#include <vector>

namespace parallax::core {
    class ExecutionContext;

    enum class SubmitResult {
        Submitted, NoWork, Failed
    };

    struct OrderedInputRequirement {
        ProductId product{};
        std::size_t history_capacity = 0;
    };
    /**
     * Contract for a computation that produces named products facing the graph.
     *
     * Producer describes dependency relationships to the Parallax graph.
     * StereoRectifier, StereoMatcher, pose estimators, or LiDAR drivers remain
     * focused on their algorithm-specific work and can be owned/referenced by a
     * producer implementation.
     *
     * Source producers are valid producers with no graph inputs.
     */
    class Producer {
        public:
            virtual ~Producer() = default;

            [[nodiscard]] virtual std::string_view name() const noexcept = 0;
            [[nodiscard]] virtual const std::vector<ProductId>& inputs() const noexcept = 0;
            [[nodiscard]] virtual const std::vector<ProductId>& outputs() const noexcept = 0;

            /**
             * Inputs requiring bounded ordered history.
             *
             * Most realtime producers consume only the latest useful product and therefore
             * return no ordered requirements. Ordered consumers opt in per input without
             * changing the graph topology represented by inputs().
             */
            [[nodiscard]] virtual const std::vector<OrderedInputRequirement>&ordered_inputs() const noexcept {
                static const std::vector<OrderedInputRequirement> none;
                return none;
            }
            // Execution characteristics are declared separately from graph dependencies.
            [[nodiscard]] virtual ExecutionPolicy execution_policy() const noexcept = 0;

            /**
             * Submit producer work using Runtime-owned shared execution resources.
             *
             * Producers continue to own/reference their algorithm-specific state.
             * The context supplies infrastructure shared across producer families:
             * product storage, execution lanes, timing, and completion dependencies.
             */
            virtual SubmitResult submit(ExecutionContext& context) = 0;
    };
}