#include <parallax/mapping/mapping_config.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>

namespace parallax::mapping {
namespace { bool positive(float v) { return std::isfinite(v) && v > 0.0F; } }

bool MappingConfig::loadFromFile(const std::filesystem::path& path) {
    try {
        const YAML::Node root = YAML::LoadFile(path.string());
        const YAML::Node m = root["mapping"] ? root["mapping"] : root;
        if (m["voxel_size_m"]) voxel_size_m = m["voxel_size_m"].as<float>();
        if (const auto n=m["depth"]) {
            if (n["min_integration_distance_m"]) min_integration_distance_m=n["min_integration_distance_m"].as<float>();
            if (n["max_integration_distance_m"]) max_integration_distance_m=n["max_integration_distance_m"].as<float>();
            if (n["integration_rate_hz"]) integration_rate_hz=n["integration_rate_hz"].as<float>();
        }
        if (const auto n=m["tsdf"]) {
            if (n["truncation_distance_vox"]) tsdf_truncation_distance_vox=n["truncation_distance_vox"].as<float>();
            if (n["max_weight"]) tsdf_max_weight=n["max_weight"].as<float>();
            if (n["visualization_rate_hz"]) tsdf_visualization_rate_hz=n["visualization_rate_hz"].as<float>();
        }
        if (const auto n=m["mesh"]) {
            if (n["update_rate_hz"]) mesh_update_rate_hz=n["update_rate_hz"].as<float>();
            if (n["min_weight"]) mesh_min_weight=n["min_weight"].as<float>();
            if (n["weld_vertices"]) mesh_weld_vertices=n["weld_vertices"].as<bool>();
        }
        if (const auto n=m["color"]) {
            if (n["enabled"]) color_enabled=n["enabled"].as<bool>();
            if (n["integration_rate_hz"]) color_integration_rate_hz=n["integration_rate_hz"].as<float>();
        }
        if (const auto n=m["debug_tsdf"]) {
            if (n["columns"]) debug_columns=n["columns"].as<std::uint32_t>();
            if (n["rows"]) debug_rows=n["rows"].as<std::uint32_t>();
            if (n["slices"]) debug_slices=n["slices"].as<std::uint32_t>();
            if (n["surface_band_m"]) debug_surface_band_m=n["surface_band_m"].as<float>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "Mapping config: " << e.what() << '\n'; return false;
    }
    if (!positive(voxel_size_m) || !positive(min_integration_distance_m) ||
        !positive(max_integration_distance_m) || max_integration_distance_m <= min_integration_distance_m ||
        !positive(integration_rate_hz) || !positive(tsdf_truncation_distance_vox) || !positive(tsdf_max_weight) ||
        !positive(tsdf_visualization_rate_hz) || !positive(mesh_update_rate_hz) || !positive(mesh_min_weight) || !positive(color_integration_rate_hz) ||
        debug_columns==0 || debug_rows==0 || debug_slices==0 || !positive(debug_surface_band_m)) {
        std::cerr << "Mapping config: invalid parameters\n"; return false;
    }
    return true;
}
}
