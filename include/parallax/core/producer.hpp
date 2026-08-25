#pragma once

#include <parallax/core/product_id.hpp>
#include <parallax/core/execution_policy.hpp>

#include <string_view>
#include <vector>

namespace parallax::core {
    class ExecutionContext;

    /**
     * intentionally small for now will add:
     * Completion events, execution resources, freshness/drop behavior, and detailed scheduling semantics
     * later
     */
    enum class SubmitResult { 
        Submitted, NoWork, Failed
    };

    /**
     * contract for a computation that produces named products facing the garph
     * 
     * Producer describes dependency relationships to the parallax graph.
     * StereoRectifier, StereoMatcher, pose estimators or LiDAR drivers remain focused on their
     * own stuff and can be own/referenced by a producer implementation
     * 
     * Source producers are valid producers with no graph inputs.
     */

     class Producer {
        public:
            virtual ~Producer() = default;
            [[nodiscard]] virtual std::string_view name() const noexcept = 0;
            [[nodiscard]] virtual const std::vector<ProductId>& inputs() const noexcept = 0;
            [[nodiscard]] virtual const std::vector<ProductId>& outputs() const noexcept = 0;

            // Execution characteristics are declared separately from graph dependencies.
            // and will eventually require explicit lifecycle handling
            [[nodiscard]] virtual ExecutionPolicy execution_policy() const noexcept = 0;

            /**
            * Submit producer work using Runtime-owned shared execution resources.
            *
            * Producers continue to own/reference their algorithm-specific state. The
            * context supplies infrastructure shared across producer families: product
            * storage, execution lanes, timing, and later completion dependencies.
            */
            virtual SubmitResult submit(ExecutionContext& context) = 0;
     };
}