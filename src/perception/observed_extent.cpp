#include <parallax/perception/observed_extent.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace parallax::perception {

    bool ObservedExtent3D::valid() const noexcept {
        return support > 0 &&
               std::isfinite(center_m[0]) && std::isfinite(center_m[1]) &&
               std::isfinite(center_m[2]) && std::isfinite(size_m[0]) &&
               std::isfinite(size_m[1]) && std::isfinite(size_m[2]) &&
               size_m[0] > 0.0F && size_m[1] > 0.0F && size_m[2] > 0.0F;
    }

    bool estimateObservedExtent(const std::vector<std::array<float, 3>>& points_m,
                                ObservedExtent3D& output,
                                float trim_fraction,
                                std::size_t min_support) noexcept {
        output = {};

        if (min_support < 2 || points_m.size() < min_support ||
            !std::isfinite(trim_fraction) || trim_fraction < 0.0F ||
            trim_fraction >= 0.5F) return false;

        std::array<std::vector<float>, 3> axes;
        for (auto& axis : axes) axis.reserve(points_m.size());

        for (const auto& point : points_m) {
            if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
                !std::isfinite(point[2]) || point[2] <= 0.0F) continue;
            axes[0].push_back(point[0]);
            axes[1].push_back(point[1]);
            axes[2].push_back(point[2]);
        }

        const std::size_t count = axes[0].size();
        if (count < min_support) return false;
        for (auto& axis : axes) std::sort(axis.begin(), axis.end());

        const std::size_t requested_trim =
            static_cast<std::size_t>(static_cast<double>(count - 1) * trim_fraction);
        const std::size_t trim = std::min(requested_trim, (count - 2) / 2);
        const std::size_t low = trim;
        const std::size_t high = count - 1 - trim;

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const float minimum = axes[axis][low];
            const float maximum = axes[axis][high];
            const float size = maximum - minimum;

            // Do not invent thickness when the measurements do not support it.
            if (!std::isfinite(size) || size <= 0.0F) {
                output = {};
                return false;
            }

            output.center_m[axis] = 0.5F * (minimum + maximum);
            output.size_m[axis] = size;
        }

        output.support = high - low + 1;
        return output.valid();
    }

}
