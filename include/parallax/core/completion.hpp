#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace parallax::core {

    enum class CompletionKind {
        None, CpuReady, Vpi, Cuda
    };

    /**
     * Identifies one specific submitted generation of accelerator work.
     *
     * This object intentionally contains no CUDA or VPI native types.
     * ExecutionContext owns the native event associated with the slot.
     *
     * The shared ticket prevents ExecutionContext from re-recording that
     * event slot while a Product generation still references it.
     */
    struct CompletionTicket {
        CompletionKind kind = CompletionKind::None;
        std::size_t slot = 0;
        std::uint64_t generation = 0;
    };

    /**
     * Lightweight readiness reference carried by a Product generation.
     *
     * A CompletionHandle does not mean that the work is already complete.
     * It identifies the exact completion primitive a consumer must depend
     * on before observing the associated payload.
     */
    struct CompletionHandle {
        CompletionKind kind = CompletionKind::None;
        std::shared_ptr<const CompletionTicket> ticket{};

        [[nodiscard]] bool valid() const noexcept {
            switch (kind) {
                case CompletionKind::CpuReady:
                    return true;

                case CompletionKind::Vpi:
                case CompletionKind::Cuda:
                    return static_cast<bool>(ticket);

                case CompletionKind::None:
                default:
                    return false;
            }
        }

        [[nodiscard]] bool requires_wait() const noexcept {
            return kind == CompletionKind::Vpi || kind == CompletionKind::Cuda;
        }

        [[nodiscard]] static CompletionHandle cpu_ready() noexcept {
            CompletionHandle handle{};
            handle.kind = CompletionKind::CpuReady;
            return handle;
        }
    };
}