#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace parallax::mapping {

    enum class OccupancyCell : std::uint8_t { Unknown = 0, Free = 1, Occupied = 2 };

    struct LocalOccupancyState {
        std::uint64_t localization_epoch = 0;
        std::uint64_t integrated_frames = 0;
        std::uint64_t epoch_resets = 0;

        std::size_t allocated_blocks = 0;
        std::size_t allocated_bytes = 0;
        
        float voxel_size_m = 0.15F;
        std::array<float, 3> origin_m{};
        
        std::uint32_t column_count = 0;
        std::uint32_t row_count = 0;
        std::uint32_t slice_count = 0;
        
        std::vector<std::uint8_t> cells;

        [[nodiscard]] bool gridValid() const noexcept {
            if (!(voxel_size_m > 0.0F) || column_count == 0 || row_count == 0 || slice_count == 0) return false;
        
            const std::size_t expected = static_cast<std::size_t>(column_count) *
                                         static_cast<std::size_t>(row_count) * 
                                         static_cast<std::size_t>(slice_count);

            return cells.size() == expected;
        }
    };
}
