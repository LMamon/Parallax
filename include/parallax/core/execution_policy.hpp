#pragma once

#include <cstdint>

namespace parallax::core {
    
    // hardware/resource affinity declared by a graph producer
    enum class ResourceAffinity : std::uint8_t {
        Cpu, Vic, Ofa, Gpu
    };

    // add Rate, freshness, priority, and drop/backpressure policy later
    struct ExecutionPolicy { 
        ResourceAffinity affinity = ResourceAffinity::Cpu;
        bool stateful = false;
    };
} // namespace parallax::core