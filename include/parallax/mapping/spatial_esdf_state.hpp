#pragma once

#include <cstddef>
#include <cstdint>

namespace parallax::mapping {

struct SpatialEsdfState {
    std::uint64_t localization_epoch = 0;
    std::uint64_t map_revision = 0;
    std::uint64_t esdf_revision = 0;
    std::size_t allocated_blocks = 0;
    float voxel_size_m = 0.0F;

    [[nodiscard]] bool valid() const noexcept {
        return localization_epoch > 0 &&
               map_revision > 0 &&
               esdf_revision > 0 &&
               voxel_size_m > 0.0F;
    }
};

}  // namespace parallax::mapping
