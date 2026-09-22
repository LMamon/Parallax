#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace parallax::mapping {
struct SpatialMeshState {
    std::uint64_t localization_epoch = 0;
    std::uint64_t integrated_frames = 0;
    std::uint64_t mesh_revision = 0;
    std::vector<std::array<float,3>> vertices_m;
    std::vector<std::array<std::uint8_t,3>> colors_rgb;
    std::vector<std::array<std::uint32_t,3>> triangles;

    [[nodiscard]] bool valid() const noexcept {
        if (vertices_m.empty() || colors_rgb.size()!=vertices_m.size()) return false;
        for (const auto& t: triangles)
            if (t[0]>=vertices_m.size() || t[1]>=vertices_m.size() || t[2]>=vertices_m.size()) return false;
        return true;
    }
};
}
