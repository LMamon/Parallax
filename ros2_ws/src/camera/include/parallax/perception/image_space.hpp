#pragma once

#include <cstdint>

namespace parallax::perception {
    enum class ImageSpace : std::uint8_t {
        Unknown = 0, RgbLeft, RectifiedLeft
    };
}