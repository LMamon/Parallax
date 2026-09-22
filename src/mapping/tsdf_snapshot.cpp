#include <parallax/mapping/tsdf_snapshot.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace parallax::mapping {
    std::uint8_t classifyTsdfVoxel(float distance_m,
                                   float weight,
                                   float surface_band_m) noexcept {
        if (!std::isfinite(distance_m) || !std::isfinite(weight) ||
            weight <= 0.0F || surface_band_m <= 0.0F) {
            return static_cast<std::uint8_t>(TsdfCell::Unknown);
        }
        return std::abs(distance_m) <= surface_band_m
            ? static_cast<std::uint8_t>(TsdfCell::Surface)
            : static_cast<std::uint8_t>(TsdfCell::Unknown);
    }

    bool buildTsdfSnapshot(const nvblox::TsdfLayer& layer,
                           const nvblox::Vector3f& center,
                           cudaStream_t stream,
                           const TsdfSnapshotConfig& config,
                           SpatialTsdfState* state) {
        if (!state || !stream || config.voxel_size_m <= 0.0F ||
            config.column_count == 0 || config.row_count == 0 ||
            config.slice_count == 0) return false;

        using Block = nvblox::VoxelBlock<nvblox::TsdfVoxel>;
        const float sx = config.column_count * config.voxel_size_m;
        const float sy = config.row_count * config.voxel_size_m;
        const float sz = config.slice_count * config.voxel_size_m;
        const auto snap = [&](float v) {
            return std::floor(v / config.voxel_size_m) * config.voxel_size_m;
        };
        const nvblox::Vector3f origin(snap(center.x() - 0.5F * sx),
                                      snap(center.y() - 0.5F * sy),
                                      snap(center.z() - 0.5F * sz));
        const nvblox::Vector3f upper(origin.x() + sx, origin.y() + sy, origin.z() + sz);
        const float block_size = layer.block_size();

        std::vector<nvblox::Index3D> selected_indices;
        std::vector<const Block*> selected_blocks;
        for (const auto& index : layer.getAllBlockIndices()) {
            const nvblox::Vector3f bmin(index.x() * block_size,
                                        index.y() * block_size,
                                        index.z() * block_size);
            const nvblox::Vector3f bmax = bmin + nvblox::Vector3f::Constant(block_size);
            const bool intersects = bmax.x() > origin.x() && bmin.x() < upper.x() &&
                                    bmax.y() > origin.y() && bmin.y() < upper.y() &&
                                    bmax.z() > origin.z() && bmin.z() < upper.z();
            if (!intersects) continue;
            auto block = layer.getBlockAtIndex(index);
            if (!block) continue;
            selected_indices.push_back(index);
            selected_blocks.push_back(block.get());
        }

        std::vector<Block> host_blocks(selected_blocks.size());
        for (std::size_t i = 0; i < selected_blocks.size(); ++i) {
            if (cudaMemcpyAsync(&host_blocks[i], selected_blocks[i], sizeof(Block),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) return false;

        state->voxel_size_m = config.voxel_size_m;
        state->origin_m = {origin.x(), origin.y(), origin.z()};
        state->column_count = config.column_count;
        state->row_count = config.row_count;
        state->slice_count = config.slice_count;
        state->cells.assign(static_cast<std::size_t>(config.column_count) *
                            config.row_count * config.slice_count,
                            static_cast<std::uint8_t>(TsdfCell::Unknown));

        for (std::size_t bi = 0; bi < host_blocks.size(); ++bi) {
            const auto& index = selected_indices[bi];
            const auto& block = host_blocks[bi];
            for (int vx = 0; vx < Block::kVoxelsPerSide; ++vx) {
                for (int vy = 0; vy < Block::kVoxelsPerSide; ++vy) {
                    for (int vz = 0; vz < Block::kVoxelsPerSide; ++vz) {
                        const float x = index.x() * block_size + (vx + 0.5F) * config.voxel_size_m;
                        const float y = index.y() * block_size + (vy + 0.5F) * config.voxel_size_m;
                        const float z = index.z() * block_size + (vz + 0.5F) * config.voxel_size_m;
                        const int gx = static_cast<int>(std::floor((x - origin.x()) / config.voxel_size_m));
                        const int gy = static_cast<int>(std::floor((y - origin.y()) / config.voxel_size_m));
                        const int gz = static_cast<int>(std::floor((z - origin.z()) / config.voxel_size_m));
                        if (gx < 0 || gy < 0 || gz < 0 ||
                            gx >= static_cast<int>(config.column_count) ||
                            gy >= static_cast<int>(config.row_count) ||
                            gz >= static_cast<int>(config.slice_count)) continue;
                        const std::size_t linear =
                            (static_cast<std::size_t>(gz) * config.row_count +
                             static_cast<std::size_t>(gy)) * config.column_count +
                            static_cast<std::size_t>(gx);
                        const auto& voxel = block.voxels[vx][vy][vz];
                        state->cells[linear] = classifyTsdfVoxel(
                            voxel.distance, voxel.weight, config.surface_band_m);
                    }
                }
            }
        }
        return state->gridValid();
    }
}
