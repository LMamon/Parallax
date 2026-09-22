#include <parallax/mapping/spatial_tsdf_state.hpp>
#include <parallax/mapping/tsdf_snapshot.hpp>

#include <gtest/gtest.h>

TEST(SpatialTsdfStateTest, DenseGridRequiresExactCellCount) {
    parallax::mapping::SpatialTsdfState state;
    state.column_count = 2;
    state.row_count = 3;
    state.slice_count = 4;
    state.cells.assign(24, 0);
    EXPECT_TRUE(state.gridValid());
    state.cells.pop_back();
    EXPECT_FALSE(state.gridValid());
}

TEST(TsdfSnapshotTest, ClassifiesOnlyObservedSurfaceBand) {
    using parallax::mapping::TsdfCell;
    using parallax::mapping::classifyTsdfVoxel;
    EXPECT_EQ(classifyTsdfVoxel(0.05F, 1.0F, 0.15F), static_cast<std::uint8_t>(TsdfCell::Surface));
    EXPECT_EQ(classifyTsdfVoxel(-0.10F, 2.0F, 0.15F), static_cast<std::uint8_t>(TsdfCell::Surface));
    EXPECT_EQ(classifyTsdfVoxel(0.30F, 1.0F, 0.15F), static_cast<std::uint8_t>(TsdfCell::Unknown));
    EXPECT_EQ(classifyTsdfVoxel(0.0F, 0.0F, 0.15F), static_cast<std::uint8_t>(TsdfCell::Unknown));
}
