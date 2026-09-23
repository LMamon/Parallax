#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace parallax::mapping {

// Lightweight publication describing a new persistent-map revision. The
// nvblox map remains owned by SpatialMap and GPU resident.
struct SpatialMapState {
    std::uint64_t localization_epoch = 0;
    std::uint64_t revision = 0;
    std::uint64_t integrated_frames = 0;
    std::uint64_t epoch_resets = 0;
    std::size_t allocated_blocks = 0;

    std::array<float, 9> world_from_camera_rotation{};
    std::array<float, 3> world_from_camera_translation_m{};
};

}  // namespace parallax::mapping
