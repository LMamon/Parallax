#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace parallax::application {

struct NavigationGoal {
    std::array<float, 3> position_m{};
    std::uint64_t revision = 0;

    [[nodiscard]] bool valid() const noexcept {
        return revision > 0 &&
               std::isfinite(position_m[0]) &&
               std::isfinite(position_m[1]) &&
               std::isfinite(position_m[2]);
    }
};

}  // namespace parallax::application
