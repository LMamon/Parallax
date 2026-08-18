#pragma once

#include <cstdint>

namespace parallax::core {
    enum class ModuleState : std::uint8_t {
        Stopped, Running, Error
    };

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