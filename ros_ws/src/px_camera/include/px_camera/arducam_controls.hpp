#pragma once

#include <cstdint>

namespace arducam {
    constexpr uint32_t EXPOSURE = 0x00980911;
    constexpr uint32_t HORIZONTAL_FLIP = 0x00980914;
    constexpr uint32_t VERTICAL_FLIP = 0x00980915;
    constexpr uint32_t TRIGGER_MODE = 0x00981901;
    constexpr uint32_t DISABLE_FRAME_TIMEOUT = 0x00981902;
    constexpr uint32_t FRAME_TIMEOUT = 0x00981903;
    constexpr uint32_t FRAME_RATE = 0x00981906;
    constexpr uint32_t ANALOG_GAIN = 0x009e0903;
}