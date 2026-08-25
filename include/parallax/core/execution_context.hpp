#pragma once

#include <parallax/core/product_store.hpp>

#include <chrono>

namespace parallax::core {

    /**
     * Runtime-scoped owner for resources shared across graph execution.
     *
     * ExecutionContext is intentionally separate from Producer and from the
     * algorithm implementations. Producers describe graph work; algorithm classes
     * continue to own algorithm-specific payloads/resources; this context owns
     * infrastructure shared by multiple producer families.
     *
     * this object grows incrementally:
     *   - completed product storage and common timing utilities
     *   - named VPI execution lanes
     *   - CUDA/TensorRT execution resources
     *   - bounded CPU execution resources
     *
     * Context lifetime is intended to match Runtime lifetime.
     */
    class ExecutionContext {
        public:
            using Clock = std::chrono::steady_clock;
            using TimePoint = Clock::time_point;

            ExecutionContext() = default;
            ~ExecutionContext() = default;

            ExecutionContext(const ExecutionContext&) = delete;
            ExecutionContext& operator=(const ExecutionContext&) = delete;

            ExecutionContext(ExecutionContext&&) = delete;
            ExecutionContext& operator=(ExecutionContext&&) = delete;

            [[nodiscard]] ProductStore& products() noexcept {
                return product_store_;
            }

            [[nodiscard]] const ProductStore& products() const noexcept {
                return product_store_;
            }

            [[nodiscard]] static TimePoint now() noexcept {
                return Clock::now();
            }

        private:
            ProductStore product_store_;
    };
}