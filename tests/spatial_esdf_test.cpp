#include <parallax/mapping/spatial_esdf_state.hpp>

#include <gtest/gtest.h>

TEST(SpatialEsdfStateTest, RequiresVersionedMappedField) {
    parallax::mapping::SpatialEsdfState state;

    EXPECT_FALSE(state.valid());

    state.localization_epoch = 2;
    state.map_revision = 8;
    state.esdf_revision = 5;
    state.allocated_blocks = 12;
    state.voxel_size_m = 0.10F;

    EXPECT_TRUE(state.valid());

    state.voxel_size_m = 0.0F;
    EXPECT_FALSE(state.valid());
}
