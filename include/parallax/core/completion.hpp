#pragma once

namespace parallax::core {

    enum class CompletionKind {
        None,
        Vpi,
        Cuda
    };

    /**
     * Non-owning reference to an accelerator completion primitive.
     *
     * ExecutionContext owns the underlying event. This handle only identifies
     * the completion dependency associated with submitted work.
     *
     * Native CUDA/VPI types intentionally do not appear here so CPU-only
     * producers do not acquire accelerator header dependencies.
     */
    struct CompletionHandle {
        CompletionKind kind = CompletionKind::None;
        void* event = nullptr;

        [[nodiscard]] bool valid() const noexcept {
            return kind != CompletionKind::None && event != nullptr;
        }
    };

} // namespace parallax::core