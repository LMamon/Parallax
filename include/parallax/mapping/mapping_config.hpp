#pragma once

#include <cstdint>
#include <filesystem>

namespace parallax::mapping {
    struct MappingConfig {
        float voxel_size_m = 0.10F;
        float min_integration_distance_m = 0.50F;
        float max_integration_distance_m = 9.00F;
        float integration_rate_hz = 10.0F;
        float tsdf_truncation_distance_vox = 4.0F;
        float tsdf_max_weight = 5.0F;
        float tsdf_visualization_rate_hz = 1.0F;
        float mesh_update_rate_hz = 2.0F;
        float mesh_min_weight = 0.10F;
        bool mesh_weld_vertices = true;
        bool color_enabled = true;
        float color_integration_rate_hz = 2.0F;
        std::uint32_t debug_columns = 80;
        std::uint32_t debug_rows = 80;
        std::uint32_t debug_slices = 40;
        float debug_surface_band_m = 0.10F;
        bool loadFromFile(const std::filesystem::path& path);
    };
}
