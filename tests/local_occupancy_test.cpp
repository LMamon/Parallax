#include <parallax/mapping/local_occupancy_producer.hpp>
#include <gtest/gtest.h>

TEST(LocalOccupancyState, DefaultsDescribeCheckpointPolicy) {
    parallax::mapping::LocalOccupancyState state{};
    EXPECT_FLOAT_EQ(state.voxel_size_m, 0.15F);
    EXPECT_EQ(state.integrated_frames, 0U);
    EXPECT_EQ(state.epoch_resets, 0U);
    EXPECT_EQ(state.allocated_blocks, 0U);
    EXPECT_EQ(state.allocated_bytes, 0U);
}
