#pragma once

#include <cstdint>

namespace parallax::core {

    /**
     * Runtime scheduling counters for one graph producer.
     *
     * These counters describe orchestration decisions, not algorithm timing.
     * They answer how often a producer was considered, executed, skipped, or
     * failed without requiring the producer implementation to know about
     * runtime scheduling.
     */
    struct ProducerExecutionStats {
        std::uint64_t considered = 0;

        std::uint64_t submitted = 0;
        std::uint64_t no_work = 0;
        std::uint64_t failed = 0;

        std::uint64_t missing_or_incompatible_input = 0;
        std::uint64_t rate_limited = 0;
        std::uint64_t stale_input = 0;
        std::uint64_t superseded = 0;
    };

} // namespace parallax::core