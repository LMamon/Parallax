#pragma once

#include <cstdint>

namespace parallax::camera::controls {

    inline constexpr std::uint32_t Exposure = 0x00980911;

    inline constexpr std::uint32_t HorizontalFlip = 0x00980914;
    inline constexpr std::uint32_t VerticalFlip = 0x00980915;

    inline constexpr std::uint32_t TriggerMode = 0x00981901;
    inline constexpr std::uint32_t DisableFrameTimeout = 0x00981902;
    inline constexpr std::uint32_t FrameTimeout = 0x00981903;

    inline constexpr std::uint32_t FrameRate = 0x00981906;

    inline constexpr std::uint32_t AnalogGain = 0x009e0903;

}