#include <parallax/mapping/spatial_mesh_snapshot.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace parallax::mapping {

bool buildSpatialMeshSnapshot(const nvblox::ColorMesh& mesh,
                              SpatialMeshState* out) {
    if (out == nullptr ||
        mesh.vertices.empty() ||
        mesh.vertex_appearances.size() != mesh.vertices.size() ||
        mesh.triangles.size() % 3U != 0U) {
        return false;
    }

    out->vertices_m.clear();
    out->colors_rgb.clear();
    out->triangles.clear();

    out->vertices_m.reserve(mesh.vertices.size());
    out->colors_rgb.reserve(mesh.vertices.size());
    out->triangles.reserve(mesh.triangles.size() / 3U);

    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const auto& vertex = mesh.vertices[i];
        const auto& color = mesh.vertex_appearances[i];

        out->vertices_m.push_back(std::array<float, 3>{
            vertex.x(), vertex.y(), vertex.z()});
        out->colors_rgb.push_back(std::array<std::uint8_t, 3>{
            color.r(), color.g(), color.b()});
    }

    for (std::size_t i = 0; i < mesh.triangles.size(); i += 3U) {
        const int index_a = mesh.triangles[i];
        const int index_b = mesh.triangles[i + 1U];
        const int index_c = mesh.triangles[i + 2U];

        if (index_a < 0 || index_b < 0 || index_c < 0) {
            return false;
        }

        const auto a = static_cast<std::uint32_t>(index_a);
        const auto b = static_cast<std::uint32_t>(index_b);
        const auto c = static_cast<std::uint32_t>(index_c);

        if (a >= out->vertices_m.size() ||
            b >= out->vertices_m.size() ||
            c >= out->vertices_m.size()) {
            return false;
        }

        out->triangles.push_back(
            std::array<std::uint32_t, 3>{a, b, c});
    }

    return out->valid();
}

}  // namespace parallax::mapping
