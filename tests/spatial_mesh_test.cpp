#include <parallax/mapping/spatial_mesh_state.hpp>
#include <gtest/gtest.h>
TEST(SpatialMeshStateTest, ValidatesColoredIndexedSurface) {
    parallax::mapping::SpatialMeshState m;
    m.vertices_m={{{0,0,0},{1,0,0},{0,1,0}}};
    m.colors_rgb={{{255,0,0},{0,255,0},{0,0,255}}};
    m.triangles={{{0,1,2}}};
    EXPECT_TRUE(m.valid());
    m.triangles[0][2]=3;
    EXPECT_FALSE(m.valid());
}
