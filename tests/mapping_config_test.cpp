#include <parallax/mapping/mapping_config.hpp>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

TEST(MappingConfigTest, LoadsMappingPolicy) {
    const auto p=std::filesystem::temp_directory_path()/"parallax_mapping_config_test.yaml";
    { std::ofstream o(p); o << R"(mapping:
  voxel_size_m: 0.075
  depth: {min_integration_distance_m: 0.6, max_integration_distance_m: 8.5, integration_rate_hz: 9.0}
  tsdf: {truncation_distance_vox: 3.0, max_weight: 4.0}
  mesh: {update_rate_hz: 1.5, min_weight: 0.2, weld_vertices: false}
  color: {enabled: false, integration_rate_hz: 1.0}
  debug_tsdf: {columns: 40, rows: 42, slices: 20, surface_band_m: 0.05}
)"; }
    parallax::mapping::MappingConfig c; ASSERT_TRUE(c.loadFromFile(p));
    EXPECT_FLOAT_EQ(c.voxel_size_m,0.075F); EXPECT_FLOAT_EQ(c.max_integration_distance_m,8.5F);
    EXPECT_FLOAT_EQ(c.tsdf_truncation_distance_vox,3.0F); EXPECT_FALSE(c.mesh_weld_vertices);
    EXPECT_FALSE(c.color_enabled); EXPECT_EQ(c.debug_rows,42U); EXPECT_FLOAT_EQ(c.debug_surface_band_m,0.05F);
    std::filesystem::remove(p);
}
