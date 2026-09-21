#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace parallax::perception {

    struct ObservedExtent3D {
        std::array<float, 3> center_m{};
        std::array<float, 3> size_m{};
        std::size_t support = 0;

        [[nodiscard]] bool valid() const noexcept;
    };

    /*
     * Fit bounds to visible stereo surface samples. Per-axis trimmed quantiles
     * prevent isolated depth failures from inflating the result.
     */
    [[nodiscard]] bool estimateObservedExtent(const std::vector<std::array<float, 3>>& points_m,
                                              ObservedExtent3D& output,
                                              float trim_fraction = 0.10F,
                                              std::size_t min_support = 8) noexcept;
}
