#pragma once

#include <parallax/mapping/spatial_tsdf_state.hpp>

#include <cuda_runtime.h>
#include <nvblox/map/common_names.h>
#include <nvblox/map/layer.h>
#include <nvblox/map/voxels.h>

namespace parallax::mapping {
    struct TsdfSnapshotConfig {
        float voxel_size_m = 0.15F;
        std::uint32_t column_count = 80;
        std::uint32_t row_count = 80;
        std::uint32_t slice_count = 40;
        float surface_band_m = 0.15F;
    };

    [[nodiscard]] std::uint8_t classifyTsdfVoxel(float distance_m,
                                                  float weight,
                                                  float surface_band_m) noexcept;

    bool buildTsdfSnapshot(const nvblox::TsdfLayer& layer,
                           const nvblox::Vector3f& center,
                           cudaStream_t stream,
                           const TsdfSnapshotConfig& config,
                           SpatialTsdfState* state);
}
