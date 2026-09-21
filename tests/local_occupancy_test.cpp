#include <parallax/mapping/local_occupancy_state.hpp>
#include <parallax/mapping/occupancy_snapshot.hpp>
#include <gtest/gtest.h>

TEST(LocalOccupancyState, DefaultsDescribeCheckpointPolicy) {
    parallax::mapping::LocalOccupancyState state{};
    EXPECT_FLOAT_EQ(state.voxel_size_m, 0.15F);
    EXPECT_EQ(state.integrated_frames, 0U);
    EXPECT_EQ(state.epoch_resets, 0U);
    EXPECT_EQ(state.allocated_blocks, 0U);
    EXPECT_EQ(state.allocated_bytes, 0U);
}

TEST(LocalOccupancyStateTest, DenseGridRequiresExactCellCount) {
    parallax::mapping::LocalOccupancyState state;
    state.column_count = 2;
    state.row_count = 3;
    state.slice_count = 4;
    state.cells.assign(
        24,
        static_cast<std::uint8_t>(parallax::mapping::OccupancyCell::Unknown));
    EXPECT_TRUE(state.gridValid());
    state.cells.pop_back();
    EXPECT_FALSE(state.gridValid());
}

TEST(LocalOccupancyStateTest, CellEncodingIsStable) {
    EXPECT_EQ(static_cast<std::uint8_t>(parallax::mapping::OccupancyCell::Unknown), 0U);
    EXPECT_EQ(static_cast<std::uint8_t>(parallax::mapping::OccupancyCell::Free), 1U);
    EXPECT_EQ(static_cast<std::uint8_t>(parallax::mapping::OccupancyCell::Occupied), 2U);
}


TEST(OccupancySnapshotTest, ClassifiesLogOddsWithoutVisualizationPolicy) {
    using parallax::mapping::OccupancyCell;
    using parallax::mapping::classifyOccupancyLogOdds;

    EXPECT_EQ(classifyOccupancyLogOdds(1.0F),
              static_cast<std::uint8_t>(OccupancyCell::Occupied));
    EXPECT_EQ(classifyOccupancyLogOdds(-1.0F),
              static_cast<std::uint8_t>(OccupancyCell::Free));
    EXPECT_EQ(classifyOccupancyLogOdds(0.0F),
              static_cast<std::uint8_t>(OccupancyCell::Unknown));
}
