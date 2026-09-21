#pragma once

#include <parallax/mapping/local_occupancy_state.hpp>

#include <cuda_runtime.h>
#include <nvblox/mapper/mapper.h>

#include <cstdint>

namespace parallax::mapping {

    struct OccupancySnapshotConfig {
        float voxel_size_m = 0.15F;

        // 8.1 m x 8.1 m x 4.05 m bounded visualization snapshot.
        std::uint32_t column_count = 54;
        std::uint32_t row_count = 54;
        std::uint32_t slice_count = 27;
    };

    [[nodiscard]] std::uint8_t classifyOccupancyLogOdds(float log_odds) noexcept;

    /*
     * Copy a bounded occupancy window into the CPU-owned product representation.
     * Mapping owns this conversion; visualization only serializes the product.
     */
    bool buildOccupancySnapshot(const nvblox::OccupancyLayer& layer,
                                const nvblox::Vector3f& center,
                                cudaStream_t stream,
                                const OccupancySnapshotConfig& config,
                                LocalOccupancyState* state);

}
