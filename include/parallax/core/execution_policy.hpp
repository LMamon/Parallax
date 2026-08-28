#pragma once

#include <cstdint>

namespace parallax::core {

    // Hardware/resource affinity declared by a graph producer.
    enum class ResourceAffinity : std::uint8_t {
        Cpu, Vic, Ofa, Gpu
    };

    // Block is the conservative default: do not discard requested work.
    // Supersede allows pending work to be replaced by newer work when only the
    // newest result remains useful.
    enum class DropPolicy : std::uint8_t {
        Block, Supersede
    };

    // Declarative execution policy attached to a producer.
    //
    // The dependency graph determines whether a producer is required and which
    // products it depends on. ExecutionPolicy describes how that required
    // producer should be paced and which inputs are acceptable.
    //
    // These fields are policy metadata only. Due-time evaluation, freshness
    // checks, and supersede behavior are implemented by the graph/runtime.
    struct ExecutionPolicy {
        // Desired maximum execution rate in Hz.
        // <= 0 means unthrottled: execute whenever demanded and otherwise valid.
        double target_hz = 0.0;

        // Maximum acceptable age of required inputs, in milliseconds.
        // <= 0 disables age-based freshness rejection.
        double max_input_age_ms = 0.0;

        // Behavior when newer work arrives while older work is still pending.
        DropPolicy drop_policy = DropPolicy::Block;

        // Relative scheduling importance. Higher values represent higher
        // importance; interpretation is left to the runtime.
        std::int32_t priority = 0;
        ResourceAffinity affinity = ResourceAffinity::Cpu;

        // Stateful producers may retain information across executions and require
        // execution ordering to be preserved.
        bool stateful = false;
    };
}