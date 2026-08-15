#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace parallax::camera {

    struct Resolution {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };

    struct RawFrame {
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t fourcc = 0;

        const void* data = nullptr;
        std::size_t bytes = 0;

        std::chrono::nanoseconds timestamp{0};

        std::uint32_t buffer_index = 0;
    };

}