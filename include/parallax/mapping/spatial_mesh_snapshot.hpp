#pragma once
#include <parallax/mapping/spatial_mesh_state.hpp>
#include <nvblox/mesh/mesh.h>
namespace parallax::mapping {
[[nodiscard]] bool buildSpatialMeshSnapshot(const nvblox::ColorMesh&, SpatialMeshState*);
}
