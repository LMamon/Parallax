#pragma once

#include <atomic>
#include <cstdint>

namespace parallax::core {

    struct ProducerExecutionStats {
        std::atomic<std::uint64_t> considered{0};

        std::atomic<std::uint64_t> submitted{0};
        std::atomic<std::uint64_t> no_work{0};
        std::atomic<std::uint64_t> failed{0};

        std::atomic<std::uint64_t> missing_or_incompatible_input{0};
        std::atomic<std::uint64_t> rate_limited{0};
        std::atomic<std::uint64_t> stale_input{0};
        std::atomic<std::uint64_t> superseded{0};
    };

}