#pragma once

#include <cstdint>

namespace parallax::core {
    enum class ModuleState : std::uint8_t {
        Stopped, Running, Error
    };

    /**
     * Legacy lifecycle abstraction.
     *
     * Module may continue to be used temporarily for start/stop ownership while
     * current refactor is in progress, but it does not describe
     * graph dependencies and must not be used as the graph execution contract.
     *
     * New graph-facing computation should implement Producer. Lifecycle
     * ownership will be migrated separately as the runtime is refactored.
     */
    class Module {
        public:
            virtual ~Module() = default;
            Module(const Module&) = delete;
            Module& operator=(const Module&) = delete;

            virtual bool start() = 0;
            virtual bool stop() = 0;

            [[nodiscard]] ModuleState state() const noexcept { return state_; }
        
        protected:
            Module() = default;
            ModuleState state_ = ModuleState::Stopped;
    };
}