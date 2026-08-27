#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace parallax::core {

    /**
     * Low-level runtime counters for operations that materially affect
     * accelerator throughput and latency.
     *
     * These counters live below individual producers because CUDA allocations,
     * memory-domain transfers, and synchronization are infrastructure behavior,
     * not graph-product semantics.
     *
     * Counters are atomic because camera, LiDAR, CPU workers, and accelerator
     * submission may execute concurrently.
     */
    struct RuntimeMetrics {
        std::atomic<std::uint64_t> cuda_allocations{0};
        std::atomic<std::uint64_t> cuda_allocated_bytes{0};
        std::atomic<std::uint64_t> cuda_frees{0};

        std::atomic<std::uint64_t> host_to_device_transfers{0};
        std::atomic<std::uint64_t> host_to_device_bytes{0};

        std::atomic<std::uint64_t> device_to_host_transfers{0};
        std::atomic<std::uint64_t> device_to_host_bytes{0};

        std::atomic<std::uint64_t> device_to_device_transfers{0};
        std::atomic<std::uint64_t> device_to_device_bytes{0};

        // Accelerator-side dependency edges. These do not block the host.
        std::atomic<std::uint64_t> accelerator_waits{0};

        // Explicit CPU observation boundaries that wait for accelerator work.
        std::atomic<std::uint64_t> host_waits{0};

        // Whole-context drains are tracked separately because they are expected
        // during shutdown and must not be confused with runtime host stalls.
        std::atomic<std::uint64_t> context_drains{0};
    };

    [[nodiscard]] inline RuntimeMetrics& runtime_metrics() noexcept {
        static RuntimeMetrics metrics;
        return metrics;
    }

    inline void reset_runtime_metrics() noexcept {
        auto& metrics = runtime_metrics();

        metrics.cuda_allocations.store(0);
        metrics.cuda_allocated_bytes.store(0);
        metrics.cuda_frees.store(0);

        metrics.host_to_device_transfers.store(0);
        metrics.host_to_device_bytes.store(0);

        metrics.device_to_host_transfers.store(0);
        metrics.device_to_host_bytes.store(0);

        metrics.device_to_device_transfers.store(0);
        metrics.device_to_device_bytes.store(0);

        metrics.accelerator_waits.store(0);
        metrics.host_waits.store(0);
        metrics.context_drains.store(0);
    }
}